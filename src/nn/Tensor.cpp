
#include "Tensor.hpp"

namespace nn {

Tensor Tensor::match(const std::string& text) {
	Tensor result;
	for (auto& textline: textlines) {
		for (auto& new_textline: textline.match(text)) {
			result.textlines.push_back(new_textline);
		}
	}
	return result;
}

Tensor Tensor::match_foreground(const stb::Image* image, const std::string& color) {
	Tensor result;
	for (auto& textline: textlines) {
		if (textline.match_foreground(image, color)) {
			result.textlines.push_back(textline);
		}
	}
	return result;
}

Tensor Tensor::match_background(const stb::Image* image, const std::string& color) {
	Tensor result;
	for (auto& textline: textlines) {
		if (textline.match_background(image, color)) {
			result.textlines.push_back(textline);
		}
	}
	return result;
}

Tensor Tensor::from_top(size_t i) const {
	if (i >= textlines.size()) {
		throw ContinueError("Can't apply specifier \"from_top\": not enough objects in the input array");
	}

	Tensor result = *this;

	std::sort(result.textlines.begin(), result.textlines.end(), [](const nn::TextLine& a, const nn::TextLine& b) {
		return a.rect.top < b.rect.top;
	});

	result.textlines = {result.textlines.at(i)};

	return result;
}

Tensor Tensor::from_bottom(size_t i) const {
	if (i >= textlines.size()) {
		throw ContinueError("Can't apply specifier \"from_bottom\": not enough objects in the input array");
	}

	Tensor result = *this;

	std::sort(result.textlines.begin(), result.textlines.end(), [](const nn::TextLine& a, const nn::TextLine& b) {
		return a.rect.bottom > b.rect.bottom;
	});

	result.textlines = {result.textlines.at(i)};

	return result;
}

Tensor Tensor::from_left(size_t i) const {
	if (i >= textlines.size()) {
		throw ContinueError("Can't apply specifier \"from_left\": not enough objects in the input array");
	}

	Tensor result = *this;

	std::sort(result.textlines.begin(), result.textlines.end(), [](const nn::TextLine& a, const nn::TextLine& b) {
		return a.rect.left < b.rect.left;
	});

	result.textlines = {result.textlines.at(i)};

	return result;
}

Tensor Tensor::from_right(size_t i) const {
	if (i >= textlines.size()) {
		throw ContinueError("Can't apply specifier \"from_right\": not enough objects in the input array");
	}

	Tensor result = *this;

	std::sort(result.textlines.begin(), result.textlines.end(), [](const nn::TextLine& a, const nn::TextLine& b) {
		return a.rect.right > b.rect.right;
	});

	result.textlines = {result.textlines.at(i)};

	return result;
}

Point Tensor::left_top() const {
	if (!textlines.size()) {
		throw ContinueError("Can't apply specifier \"left_top\": there's no input object");
	}

	if (textlines.size() > 1) {
		throw std::runtime_error("Can't apply specifier \"left_top\": there's more than one object");
	}

	return textlines.at(0).rect.left_top();
}

Point Tensor::left_bottom() const {
	if (!textlines.size()) {
		throw ContinueError("Can't apply specifier \"left_bottom\": there's no input object");
	}

	if (textlines.size() > 1) {
		throw std::runtime_error("Can't apply specifier \"left_bottom\": there's more than one object");
	}

	return textlines.at(0).rect.left_bottom();
}

Point Tensor::right_top() const {
	if (!textlines.size()) {
		throw ContinueError("Can't apply specifier \"right_top\": there's no input object");
	}

	if (textlines.size() > 1) {
		throw std::runtime_error("Can't apply specifier \"right_top\": there's more than one object");
	}

	return textlines.at(0).rect.right_top();
}

Point Tensor::right_bottom() const {
	if (!textlines.size()) {
		throw ContinueError("Can't apply specifier \"right_bottom\": there's no input object");
	}

	if (textlines.size() > 1) {
		throw std::runtime_error("Can't apply specifier \"right_bottom\": there's more than one object");
	}

	return textlines.at(0).rect.right_bottom();
}

Point Tensor::center() const {
	if (!textlines.size()) {
		throw ContinueError("Can't apply specifier \"center\": there's no input object");
	}

	if (textlines.size() > 1) {
		throw std::runtime_error("Can't apply specifier \"center\": there's more than one object");
	}

	return textlines.at(0).rect.center();
}

Point Tensor::center_top() const {
	if (!textlines.size()) {
		throw ContinueError("Can't apply specifier \"center_top\": there's no input object");
	}

	if (textlines.size() > 1) {
		throw std::runtime_error("Can't apply specifier \"center_top\": there's more than one object");
	}

	return textlines.at(0).rect.center_top();
}

Point Tensor::center_bottom() const {
	if (!textlines.size()) {
		throw ContinueError("Can't apply specifier \"center_bottom\": there's no input object");
	}

	if (textlines.size() > 1) {
		throw std::runtime_error("Can't apply specifier \"center_bottom\": there's more than one object");
	}

	return textlines.at(0).rect.center_bottom();
}

Point Tensor::left_center() const {
	if (!textlines.size()) {
		throw ContinueError("Can't apply specifier \"left_center\": there's no input object");
	}

	if (textlines.size() > 1) {
		throw std::runtime_error("Can't apply specifier \"left_center\": there's more than one object");
	}

	return textlines.at(0).rect.left_center();
}

Point Tensor::right_center() const {
	if (!textlines.size()) {
		throw ContinueError("Can't apply specifier \"right_center\": there's no input object");
	}

	if (textlines.size() > 1) {
		throw std::runtime_error("Can't apply specifier \"right_center\": there's more than one object");
	}

	return textlines.at(0).rect.right_center();
}

}
