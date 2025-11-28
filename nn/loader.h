#pragma once

#include "../core/tensor.h"
#include <string>
#include <vector>

namespace fastserve {

struct LabeledImage {
    int label;
    Tensor3D pixels;
};

std::vector<LabeledImage> load_test_images(const std::string& filepath, int max_images = -1);

void print_ascii_image(const Tensor3D& image);

}
