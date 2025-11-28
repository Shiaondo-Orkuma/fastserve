#include "loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace fastserve {

std::vector<LabeledImage> load_test_images(const std::string& filepath, int max_images) {
    std::vector<LabeledImage> images;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open test images file: " << filepath << "\n";
        return images;
    }
    
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.find("NUM_IMAGES") != std::string::npos) break;
    }
    
    int num_images;
    std::istringstream header(line);
    std::string token;
    header >> token >> num_images;
    
    if (max_images > 0) {
        num_images = std::min(num_images, max_images);
    }
    
    for (int img_idx = 0; img_idx < num_images; img_idx++) {
        LabeledImage img;
        img.pixels.resize(1, std::vector<std::vector<float>>(28, std::vector<float>(28)));
        
        while (std::getline(file, line)) {
            if (line.find("IMAGE") != std::string::npos) break;
        }
        
        std::getline(file, line);
        std::istringstream label_line(line);
        label_line >> token >> img.label;
        
        for (int row = 0; row < 28; row++) {
            std::getline(file, line);
            std::istringstream pixel_line(line);
            for (int col = 0; col < 28; col++) {
                pixel_line >> img.pixels[0][row][col];
            }
        }
        
        images.push_back(img);
    }
    
    file.close();
    return images;
}

void print_ascii_image(const Tensor3D& img) {
    for (int row = 0; row < 28; row += 2) {
        std::cout << "    ";
        for (int col = 0; col < 28; col += 2) {
            float val = (img[0][row][col] + img[0][row+1][col] + 
                        img[0][row][col+1] + img[0][row+1][col+1]) / 4.0f;
            if (val > 0.5f) std::cout << "██";
            else if (val > 0.25f) std::cout << "▓▓";
            else if (val > 0.1f) std::cout << "░░";
            else std::cout << "  ";
        }
        std::cout << "\n";
    }
}

}
