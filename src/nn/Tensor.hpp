
#pragma once

#include <stdexcept>
#include <vector>
#include <algorithm>

#include "nlohmann/json.hpp"

#include "Point.hpp"


namespace nn {

struct ContinueError: std::runtime_error {
	using std::runtime_error::runtime_error;
};

struct LogicError: std::runtime_error {
	using std::runtime_error::runtime_error;
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
			throw LogicError("Can't get property \"x\": there's more than one object");
		}

		return objects.at(0).rect.center_x();
	}

	int32_t y() const {
		if (!objects.size()) {
			throw ContinueError("Can't get property \"y\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't get property \"y\": there's more than one object");
		}

		return objects.at(0).rect.center_y();
	}

	Point left_top() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"left_top\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't apply specifier \"left_top\": there's more than one object");
		}

		return objects.at(0).rect.left_top();
	}

	Point left_bottom() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"left_bottom\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't apply specifier \"left_bottom\": there's more than one object");
		}

		return objects.at(0).rect.left_bottom();
	}

	Point right_top() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"right_top\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't apply specifier \"right_top\": there's more than one object");
		}

		return objects.at(0).rect.right_top();
	}

	Point right_bottom() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"right_bottom\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't apply specifier \"right_bottom\": there's more than one object");
		}

		return objects.at(0).rect.right_bottom();
	}

	Point center() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"center\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't apply specifier \"center\": there's more than one object");
		}

		return objects.at(0).rect.center();
	}

	Point center_top() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"center_top\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't apply specifier \"center_top\": there's more than one object");
		}

		return objects.at(0).rect.center_top();
	}

	Point center_bottom() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"center_bottom\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't apply specifier \"center_bottom\": there's more than one object");
		}

		return objects.at(0).rect.center_bottom();
	}

	Point left_center() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"left_center\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't apply specifier \"left_center\": there's more than one object");
		}

		return objects.at(0).rect.left_center();
	}

	Point right_center() const {
		if (!objects.size()) {
			throw ContinueError("Can't apply specifier \"right_center\": there's no input object");
		}

		if (objects.size() > 1) {
			throw LogicError("Can't apply specifier \"right_center\": there's more than one object");
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

template <typename Object>
void to_json(nlohmann::json& j, const nn::Tensor<Object>& tensor) {
	j = nlohmann::json::array();

	for (auto& obj: tensor.objects) {
		j.push_back(obj);
	}
}

template <typename Object>
void from_json(const nlohmann::json& j, nn::Tensor<Object>& tensor) {
	for (auto& i: j) {
		tensor.objects.push_back(i.get<Object>());
	}
}

}
