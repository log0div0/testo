
#pragma once

#include "Tensor.hpp"
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace nn {

Tensor find_text(const stb::Image<stb::RGB>* image);
Tensor find_img(const stb::Image<stb::RGB>* image, const fs::path& path_to_img);

}
