
#pragma once

#include "TextLine.hpp"
#include <stdexcept>

namespace nn {

struct ContinueError: std::runtime_error {
	ContinueError(const char* reason): std::runtime_error(reason) {}
	ContinueError(const std::string& reason): ContinueError(reason.c_str()) {}
};

template <typename Object>
struct Tensor {
	std::vector<Object> objects;

	size_t size() const {
		return objects.size();
	}

	int32_t x() const {
		if (!objects.size()) {
			throw ContinueError("Can't get property \"x\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't get property \"x\": there's more than one object");
		}

		return objects.at(0).rect.center_x();
	}

	int32_t y() const {
		if (!objects.size()) {
			throw ContinueError("Can't get property \"y\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't get property \"y\": there's more than one object");
		}

		return objects.at(0).rect.center_y();
	}

	Point left_top() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"left_top\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't apply specifier \"left_top\": there's more than one object");
		}

		return objects.at(0).rect.left_top();
	}

	Point left_bottom() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"left_bottom\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't apply specifier \"left_bottom\": there's more than one object");
		}

		return objects.at(0).rect.left_bottom();
	}

	Point right_top() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"right_top\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't apply specifier \"right_top\": there's more than one object");
		}

		return objects.at(0).rect.right_top();
	}

	Point right_bottom() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"right_bottom\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't apply specifier \"right_bottom\": there's more than one object");
		}

		return objects.at(0).rect.right_bottom();
	}

	Point center() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"center\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't apply specifier \"center\": there's more than one object");
		}

		return objects.at(0).rect.center();
	}

	Point center_top() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"center_top\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't apply specifier \"center_top\": there's more than one object");
		}

		return objects.at(0).rect.center_top();
	}

	Point center_bottom() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"center_bottom\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't apply specifier \"center_bottom\": there's more than one object");
		}

		return objects.at(0).rect.center_bottom();
	}

	Point left_center() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"left_center\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't apply specifier \"left_center\": there's more than one object");
		}

		return objects.at(0).rect.left_center();
	}

	Point right_center() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"right_center\": there's no input object");
		}

		if (objects.size() > 1) {
			throw std::runtime_error("Can't apply specifier \"right_center\": there's more than one object");
		}

		return objects.at(0).rect.right_center();
	}
};

template <typename TensorType>
TensorType from_top(const TensorType& tensor, size_t i) {
	if (i >= tensor.objects.size()) {
		throw ContinueError("Can't apply specifier \"from_top\": not enough objects in the input array");
	}

	TensorType result = tensor;

	std::sort(result.objects.begin(), result.objects.end(), [](auto& a, auto& b) {
		return a.rect.top < b.rect.top;
	});

	result.objects = {result.objects.at(i)};

	return result;
}

template <typename TensorType>
TensorType from_bottom(const TensorType& tensor, size_t i) {
	if (i >= tensor.objects.size()) {
		throw ContinueError("Can't apply specifier \"from_bottom\": not enough objects in the input array");
	}

	TensorType result = tensor;

	std::sort(result.objects.begin(), result.objects.end(), [](auto& a, auto& b) {
		return a.rect.bottom > b.rect.bottom;
	});

	result.objects = {result.objects.at(i)};

	return result;
}

template <typename TensorType>
TensorType from_left(const TensorType& tensor, size_t i) {
	if (i >= tensor.objects.size()) {
		throw ContinueError("Can't apply specifier \"from_left\": not enough objects in the input array");
	}

	TensorType result = tensor;

	std::sort(result.objects.begin(), result.objects.end(), [](auto& a, auto& b) {
		return a.rect.left < b.rect.left;
	});

	result.objects = {result.objects.at(i)};

	return result;
}

template <typename TensorType>
TensorType from_right(const TensorType& tensor, size_t i) {
	if (i >= tensor.objects.size()) {
		throw ContinueError("Can't apply specifier \"from_right\": not enough objects in the input array");
	}

	TensorType result = tensor;

	std::sort(result.objects.begin(), result.objects.end(), [](auto& a, auto& b) {
		return a.rect.right > b.rect.right;
	});

	result.objects = {result.objects.at(i)};

	return result;
}

struct TextTensor: Tensor<TextLine> {
	TextTensor from_left(size_t i) const { return nn::from_left(*this, i); }
	TextTensor from_top(size_t i) const { return nn::from_top(*this, i); }
	TextTensor from_right(size_t i) const { return nn::from_right(*this, i); }
	TextTensor from_bottom(size_t i) const { return nn::from_bottom(*this, i); }

	TextTensor match(const std::string& text);
	TextTensor match_foreground(const stb::Image<stb::RGB>* image, const std::string& color);
	TextTensor match_background(const stb::Image<stb::RGB>* image, const std::string& color);
};

struct Img {
	Rect rect;
};

struct ImgTensor: Tensor<Img> {
	ImgTensor from_left(size_t i) const { return nn::from_left(*this, i); }
	ImgTensor from_top(size_t i) const { return nn::from_top(*this, i); }
	ImgTensor from_right(size_t i) const { return nn::from_right(*this, i); }
	ImgTensor from_bottom(size_t i) const { return nn::from_bottom(*this, i); }

};

}
