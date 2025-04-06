#include <iostream>
#include <string>
#include <cstdarg>
#include <filesystem>
#include "Logger.h"

namespace trt_log {

    static string file_name(const string &path, bool include_suffix) {
        if (path.empty()) return "";

        int p = path.rfind('/');
        int e = path.rfind('\\');
        p = max(p, e);
        p += 1;

        // include suffix
        if (include_suffix) return path.substr(p);

        int u = path.rfind('.');
        if (u == -1) return path.substr(p);

        if (u <= p) u = path.size();
        return path.substr(p, u - p);
    }

    void _log_func(const char *file, int line, const char *fmt, ...) {
        va_list vl;
        va_start(vl, fmt);
        char buffer[2048];
        string filename = file_name(file, true);
        int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
        vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
        fprintf(stdout, "%s\n", buffer);
        // 检查目录是否存在
        string folder_path = "trt_log"; // 文件夹路径
        if (!filesystem::exists(folder_path)) {
            // 如果文件夹不存在，创建文件夹
            try {
                filesystem::create_directory(folder_path);
            } catch (const exception &e) {
                cerr << "创建文件夹时出错: " << e.what() << endl;
            }
        }
        // 打开文件并追加日志
        FILE *log_file = fopen("trt_log/log.txt", "a"); // 以追加模式打开 log.txt
        if (log_file != nullptr) {
            fprintf(log_file, "%s\n", buffer); // 将日志写入文件
            fclose(log_file); // 关闭文件
        }
        va_end(vl);
    }
}
