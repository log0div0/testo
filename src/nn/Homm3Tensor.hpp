
#pragma once

#include "Tensor.hpp"
#include <stb/Image.hpp>
#include "Homm3Object.hpp"

#include <ghc/filesystem.hpp>

namespace nn {

struct Homm3Tensor: Tensor<Homm3Object> {
	Homm3Tensor from_left(size_t i) const { return nn::from_left(*this, i); }
	Homm3Tensor from_top(size_t i) const { return nn::from_top(*this, i); }
	Homm3Tensor from_right(size_t i) const { return nn::from_right(*this, i); }
	Homm3Tensor from_bottom(size_t i) const { return nn::from_bottom(*this, i); }

	Homm3Tensor match_class(const stb::Image<stb::RGB>* image, const std::string& class_id);
};

Homm3Tensor find_homm3(const stb::Image<stb::RGB>* image);

}
