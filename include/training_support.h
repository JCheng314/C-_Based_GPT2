#ifndef TRAINING_SUPPORT_H
#define TRAINING_SUPPORT_H

#include <cstdint>
#include <cstdlib>
#include <climits>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct TrainingRuntimeConfig {
    int max_steps = 2000;
    int eval_every = 100;
    int eval_iters = 10;
    int save_every = 0;
    std::string checkpoint_path;
    bool resume = false;
    bool require_cuda_aware_mpi = true;
};

inline void print_runtime_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " train.bin valid.bin [options]\n"
        << "Options:\n"
        << "  --max-steps N\n"
        << "  --eval-every N\n"
        << "  --eval-iters N\n"
        << "  --save-every N\n"
        << "  --checkpoint PATH\n"
        << "  --resume\n"
        << "  --allow-host-mpi-staging\n";
}

inline int parse_positive_int(const char* value, const char* name) {
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    }
    return static_cast<int>(parsed);
}

inline TrainingRuntimeConfig parse_runtime_config(int argc, char** argv, int first_option) {
    TrainingRuntimeConfig cfg;
    for (int i = first_option; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--max-steps") {
            cfg.max_steps = parse_positive_int(require_value("--max-steps"), "--max-steps");
        } else if (arg == "--eval-every") {
            cfg.eval_every = parse_positive_int(require_value("--eval-every"), "--eval-every");
        } else if (arg == "--eval-iters") {
            cfg.eval_iters = parse_positive_int(require_value("--eval-iters"), "--eval-iters");
        } else if (arg == "--save-every") {
            cfg.save_every = parse_positive_int(require_value("--save-every"), "--save-every");
        } else if (arg == "--checkpoint") {
            cfg.checkpoint_path = require_value("--checkpoint");
        } else if (arg == "--resume") {
            cfg.resume = true;
        } else if (arg == "--allow-host-mpi-staging") {
            cfg.require_cuda_aware_mpi = false;
        } else if (arg == "--help" || arg == "-h") {
            print_runtime_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }
    if ((cfg.save_every > 0 || cfg.resume) && cfg.checkpoint_path.empty()) {
        throw std::runtime_error("--checkpoint is required with --save-every or --resume");
    }
    return cfg;
}

inline std::vector<uint32_t> load_tokens_u32_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0) {
        throw std::runtime_error("Token file must contain uint32 little-endian token ids: " + filename);
    }

    std::vector<uint32_t> tokens(size / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char*>(tokens.data()), size)) {
        throw std::runtime_error("Failed to read file: " + filename);
    }
    return tokens;
}

struct DeviceTensorRef {
    const char* name;
    float* ptr;
    int count;
};

inline void write_or_throw(std::ofstream& out, const void* data, std::streamsize bytes) {
    if (!out.write(reinterpret_cast<const char*>(data), bytes)) {
        throw std::runtime_error("Failed while writing checkpoint");
    }
}

inline void read_or_throw(std::ifstream& in, void* data, std::streamsize bytes) {
    if (!in.read(reinterpret_cast<char*>(data), bytes)) {
        throw std::runtime_error("Failed while reading checkpoint");
    }
}

#endif // TRAINING_SUPPORT_H
