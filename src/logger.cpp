#include <iostream>
#include <string>
#include <cstdarg>
#include <filesystem>
#include "logger.h"

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
        string folder_path = "trt_log";
        if (!filesystem::exists(folder_path)) {
            try {
                filesystem::create_directory(folder_path);
            } catch (const exception &e) {
                cerr << "create_directory trt_log error" << e.what() << endl;
            }
        }
        FILE *log_file = fopen("trt_log/log.txt", "a");
        if (log_file != nullptr) {
            fprintf(log_file, "%s\n", buffer);
            fclose(log_file);
        }
        va_end(vl);
    }
}
