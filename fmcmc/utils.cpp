#include "utils.h"
#include "fmcmc.h"

int get_msb(int x) {
    return (1 << (31 - __builtin_clz(x)));
}

void Logger::store(std::string var_name, double x) {
    if (ht.find(var_name) == ht.end()) ht[var_name] = std::vector<double>();
    ht[var_name].push_back(x);
}

Logger::~Logger() {
    fout << "{";
    int i = 0;
    for (const auto& [var_name, arr]: ht) {
        fout << (i++ > 0? ",\n": "\n") << "\t\"" << var_name << "\": [";
        int j = 0;
        for (double x: arr) fout << (j++ > 0? ", ": "") << x;
        fout << "]";
    }
    fout << "\n}\n";
}
