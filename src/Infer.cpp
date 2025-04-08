#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cstdarg>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include "Logger.h"
#include "Infer.h"

#include <cassert>
#include <cstring>

namespace trt {
    using namespace std;
    using namespace nvinfer1;

    static string format_shape(const Dims &shape) {
        stringstream output;
        char buf[64];
        const char *fmts[] = {"%d", "x%d"};
        for (int i = 0; i < shape.nbDims; ++i) {
            snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
            output << buf;
        }
        return output.str();
    }

    class native_nvinfer_logger : public ILogger {
    public:
        void log(Severity severity, const char *msg) noexcept override {
            if (severity == Severity::kINTERNAL_ERROR) {
                INFO("NVInfer INTERNAL_ERROR: %s", msg);
                abort();
            } else if (severity == Severity::kERROR) {
                INFO("NVInfer: %s", msg);
            } else if (severity == Severity::kWARNING) {
                INFO("NVInfer: %s", msg);
            } else if (severity == Severity::kINFO) {
                INFO("NVInfer: %s", msg);
            } else {
                INFO("%s", msg);
            }
        }
    };

    static native_nvinfer_logger gLogger;

    template<typename T>
    static void destroy_nvidia_pointer(T *ptr) {
        delete ptr;
    }

    static vector<uint8_t> load_file(const string &file) {
        ifstream in(file, ios::in | ios::binary);
        if (!in.is_open()) return {};

        in.seekg(0, ios::end);
        size_t length = in.tellg();

        vector<uint8_t> data;
        if (length > 0) {
            in.seekg(0, ios::beg);
            data.resize(length);

            in.read(reinterpret_cast<char *>(&data[0]), length);
        }
        in.close();
        return data;
    }

    class native_engine_context {
    public:
        shared_ptr<IExecutionContext> context_;
        shared_ptr<ICudaEngine> engine_;
        shared_ptr<IRuntime> runtime_ = nullptr;

        virtual ~native_engine_context() { destroy(); }

        bool construct(const void *pdata, size_t size) {
            destroy();

            if (pdata == nullptr || size == 0) return false;
            runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
            if (runtime_ == nullptr) return false;

            engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size),
                                              destroy_nvidia_pointer<ICudaEngine>);
            if (engine_ == nullptr) return false;

            context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(),
                                                     destroy_nvidia_pointer<IExecutionContext>);
            return context_ != nullptr;
        }

    private:
        void destroy() {
            INFO("TensorRT Destroy");
            context_.reset();
            engine_.reset();
            runtime_.reset();
        }
    };

    class InferImpl : public Infer {
    public:
        shared_ptr<native_engine_context> context_;
        unordered_map<int, string> binding_index_to_name_;

        virtual ~InferImpl() = default;

        bool construct(const void *data, size_t size) {
            context_ = make_shared<native_engine_context>();
            if (!context_->construct(data, size)) {
                return false;
            }

            setup();
            return true;
        }

        bool load(const string &file) {
            auto data = load_file(file);
            if (data.empty()) {
                INFO("An empty file has been loaded. Please confirm your file path: %s", file.c_str());
                return false;
            }
            return this->construct(data.data(), data.size());
        }

        void setup() {
            auto engine = this->context_->engine_;
            int nbBindings = engine->getNbIOTensors();

            binding_index_to_name_.clear();
            for (int i = 0; i < nbBindings; ++i) {
                const char *bindingName = engine->getIOTensorName(i);
                binding_index_to_name_[i] = bindingName;
            }
        }

        virtual string name(int index) override {
            auto iter = binding_index_to_name_.find(index);
            Assertf(iter != binding_index_to_name_.end(), "Can not found the binding i: %i",
                    index);
            return iter->second;
        }

        virtual bool forward(const vector<void *> &bindings, void *stream) override {
            auto inputName = binding_index_to_name_[0];
            auto outputName = binding_index_to_name_[1];
            this->context_->context_->setTensorAddress(inputName.c_str(), bindings[0]);
            this->context_->context_->setTensorAddress(outputName.c_str(), bindings[1]);
            return this->context_->context_->enqueueV3(static_cast<cudaStream_t>(stream));
        }

        virtual vector<int> run_dims(const string &name) override {
            auto dim = this->context_->context_->getTensorShape(name.c_str());
            return vector<int>(dim.d, dim.d + dim.nbDims);
        }

        virtual vector<int> static_dims(const string &name) override {
            auto dim = this->context_->engine_->getTensorShape(name.c_str());
            return vector<int>(dim.d, dim.d + dim.nbDims);
        }

        virtual int num_bindings() override { return this->context_->engine_->getNbIOTensors(); }

        virtual bool is_input(const string &name) override {
            return this->context_->engine_->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT;
        }

        virtual bool set_run_dims(const string &name, const vector<int> &dims) override {
            Dims d;
            for (int i = 0; i < dims.size(); ++i) {
                d.d[i] = dims[i];
            }
            d.nbDims = dims.size();
            return this->context_->context_->setInputShape(name.c_str(), d);
        }

        virtual int numel(const string &name) override {
            auto dim = this->context_->context_->getTensorShape(name.c_str());
            return accumulate(dim.d, dim.d + dim.nbDims, 1, multiplies<int>());
        }

        virtual DType dtype(const string &name) override {
            return static_cast<DType>(this->context_->engine_->getTensorDataType(name.c_str()));
        }

        virtual bool has_dynamic_dim() override {
            int numBindings = this->context_->engine_->getNbIOTensors();
            for (int i = 0; i < numBindings; ++i) {
                const char *bindingName = this->context_->engine_->getIOTensorName(i);
                Dims dims = this->context_->engine_->getTensorShape(bindingName);
                for (int j = 0; j < dims.nbDims; ++j) {
                    if (dims.d[j] == -1) return true;
                }
            }
            return false;
        }

        virtual void print() override {
            INFO("Infer %p [%s]", this, has_dynamic_dim() ? "DynamicShape" : "StaticShape");

            int num_input = 0;
            int num_output = 0;
            auto engine = this->context_->engine_;
            for (int i = 0; i < engine->getNbIOTensors(); ++i) {
                string name = engine->getIOTensorName(i);
                if (engine->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT)
                    num_input++;
                else
                    num_output++;
            }

            INFO("Inputs: %d", num_input);
            for (int i = 0; i < num_input; ++i) {
                auto name = engine->getIOTensorName(i);
                auto dim = engine->getTensorShape(name);
                INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
            }

            INFO("Outputs: %d", num_output);
            for (int i = 0; i < num_output; ++i) {
                auto name = engine->getIOTensorName(i + num_input);
                auto dim = engine->getTensorShape(name);
                INFO("\t%d.%s : shape {%s}", i, name, format_shape(dim).c_str());
            }
        }
    };

    shared_ptr<Infer> load(const string &file) {
        auto *impl = new InferImpl();
        if (!impl->load(file)) {
            delete impl;
            impl = nullptr;
        }
        return shared_ptr<InferImpl>(impl);
    }

    string format_shape(const vector<int> &shape) {
        stringstream output;
        char buf[64];
        const char *fmts[] = {"%d", "x%d"};
        for (int i = 0; i < (int) shape.size(); ++i) {
            snprintf(buf, sizeof(buf), fmts[i != 0], shape[i]);
            output << buf;
        }
        return output.str();
    }
}; // namespace trt
