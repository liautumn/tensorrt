#ifndef INFER_H
#define INFER_H

#include <memory>
#include <string>
#include <vector>

namespace trt {

    using namespace std;

    enum class DType : int {
        FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3, BOOL = 4, UINT8 = 5
    };

    class infer {
    public:
        virtual bool forward(const vector<void *> &bindings, int n, void *stream = nullptr) = 0;

        virtual string name(int index) = 0;

        virtual vector<int> run_dims(const string &name) = 0;

        virtual vector<int> static_dims(const string &name) = 0;

        virtual int numel(const string &name) = 0;

        virtual int num_bindings() = 0;

        virtual bool is_input(const string &name) = 0;

        virtual bool set_run_dims(const string &name, const vector<int> &dims) = 0;

        virtual DType dtype(const string &name) = 0;

        virtual bool has_dynamic_dim() = 0;

        virtual void print() = 0;
    };

    shared_ptr<infer> load(const string &file);

    string format_shape(const vector<int> &shape);
} // namespace trt

#endif  // INFER_H
