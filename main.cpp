#include "nn/model.h"
#include "nn/loader.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>

using namespace fastserve;

void print_banner() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  FastServe v0.1 - Neural Network Inference Engine            ║\n";
    std::cout << "║  Built from scratch with SIMD-optimized kernels              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
}

void print_help() {
    print_banner();
    std::cout << "USAGE:\n";
    std::cout << "  fastserve [OPTIONS]\n\n";
    std::cout << "OPTIONS:\n";
    std::cout << "  --model <path>    Path to model weights file (required)\n";
    std::cout << "  --images <path>   Path to test images file\n";
    std::cout << "  --benchmark       Run inference benchmark\n";
    std::cout << "  --verbose         Show detailed output\n";
    std::cout << "  --help            Show this help message\n";
    std::cout << "\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  # Run inference on test images\n";
    std::cout << "  ./fastserve --model models/tiny_cnn_weights.txt --images models/test_images.txt\n";
    std::cout << "\n";
    std::cout << "  # Run benchmark\n";
    std::cout << "  ./fastserve --model models/tiny_cnn_weights.txt --benchmark\n";
    std::cout << "\n";
}

void run_inference(const TinyCNN& model, const std::string& images_path, bool verbose) {
    std::cout << "Loading images from: " << images_path << "\n";
    
    auto images = load_test_images(images_path);
    if (images.empty()) {
        std::cerr << "Error: No images loaded!\n";
        return;
    }
    
    std::cout << "Loaded " << images.size() << " images\n\n";
    
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "                    INFERENCE RESULTS\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    
    int correct = 0;
    long total_time_us = 0;
    
    for (size_t i = 0; i < images.size(); i++) {
        const auto& img = images[i];
        
        std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "│ Image " << i << " - True label: " << img.label 
                  << "                                      │\n";
        std::cout << "└─────────────────────────────────────────────────────────────┘\n";
        
        if (verbose) {
            print_ascii_image(img.pixels);
            std::cout << "\n";
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = model.predict_full(img.pixels);
        auto end = std::chrono::high_resolution_clock::now();
        auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        total_time_us += time_us;
        
        bool is_correct = (result.predicted_class == img.label);
        if (is_correct) correct++;
        
        if (verbose) {
            std::cout << "    Probabilities: ";
            for (int j = 0; j < 10; j++) {
                std::cout << j << ":" << std::fixed << std::setprecision(1) 
                          << (result.probabilities[j] * 100) << "% ";
            }
            std::cout << "\n";
        }
        
        std::cout << "   Prediction: " << result.predicted_class 
                  << " (confidence: " << std::fixed << std::setprecision(1) 
                  << (result.confidence * 100) << "%)";
        
        if (is_correct) {
            std::cout << "  CORRECT\n";
        } else {
            std::cout << "  WRONG (true: " << img.label << ")\n";
        }
        std::cout << "      Time: " << time_us << " μs\n\n";
    }
    
    // Summary
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "                        SUMMARY\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    
    float accuracy = 100.0f * correct / images.size();
    std::cout << "  Accuracy:     " << correct << "/" << images.size() 
              << " (" << std::fixed << std::setprecision(1) << accuracy << "%)\n";
    std::cout << "  Avg latency:  " << (total_time_us / images.size()) << " μs/image\n";
    std::cout << "  Throughput:   " << std::fixed << std::setprecision(0) 
              << (1000000.0 * images.size() / total_time_us) << " images/sec\n\n";
}

void run_benchmark(const TinyCNN& model) {
    std::cout << "🏃 Running benchmark...\n\n";
    
    Tensor3D dummy_input(1, std::vector<std::vector<float>>(28, std::vector<float>(28, 0.5f)));
    
    std::cout << "  Warmup (100 iterations)...\n";
    for (int i = 0; i < 100; i++) {
        model.predict(dummy_input);
    }
    
    int iterations = 1000;
    std::cout << "  Benchmarking (" << iterations << " iterations)...\n\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        model.predict(dummy_input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_us = static_cast<double>(total_us) / iterations;
    double throughput = 1000000.0 / avg_us;
    
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "                    BENCHMARK RESULTS\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";
    
    std::cout << "  Model:        TinyCNN (" << model.num_parameters() << " parameters)\n";
    std::cout << "  Architecture: " << model.architecture_string() << "\n\n";
    
    std::cout << "  Iterations:   " << iterations << "\n";
    std::cout << "  Total time:   " << std::fixed << std::setprecision(2) 
              << (total_us / 1000.0) << " ms\n";
    std::cout << "  Avg latency:  " << std::fixed << std::setprecision(1) 
              << avg_us << " μs/image\n";
    std::cout << "  Throughput:   " << std::fixed << std::setprecision(0) 
              << throughput << " images/sec\n\n";
    
    // Performance tier
    std::cout << "  Performance:  ";
    if (throughput > 50000) {
        std::cout << "🚀 Excellent (>50k img/s)\n";
    } else if (throughput > 20000) {
        std::cout << "⚡ Great (>20k img/s)\n";
    } else if (throughput > 10000) {
        std::cout << "✓ Good (>10k img/s)\n";
    } else {
        std::cout << "📈 Room for improvement\n";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    std::string model_path;
    std::string images_path;
    bool benchmark = false;
    bool verbose = false;
    bool show_help = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--images") == 0 && i + 1 < argc) {
            images_path = argv[++i];
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            show_help = true;
        }
    }
    
    if (show_help || argc == 1) {
        print_help();
        return 0;
    }
    
    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        std::cerr << "Use --help for usage information\n";
        return 1;
    }
    
    if (!benchmark && images_path.empty()) {
        std::cerr << "Error: Either --images or --benchmark is required\n";
        std::cerr << "Use --help for usage information\n";
        return 1;
    }
    
    print_banner();
    
    std::cout << "📂 Loading model from: " << model_path << "\n";
    TinyCNN model;
    if (!model.load(model_path)) {
        std::cerr << "Error: Failed to load model\n";
        return 1;
    }
    std::cout << "✓ Model loaded (" << model.num_parameters() << " parameters)\n";
    std::cout << "  Architecture: " << model.architecture_string() << "\n\n";
    
    if (benchmark) {
        run_benchmark(model);
    } else {
        run_inference(model, images_path, verbose);
    }
    
    return 0;
}
