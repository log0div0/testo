
#pragma once

#include "Network.hpp"
#include "Image.hpp"
#include <fstream>
#include <inipp.hh>

namespace yolo {

struct Box {
	float x, y, h, w;
	float uio(const Box& other) const {
		return intersection(other)/union_(other);
	}
	float intersection(const Box& other) const {
		float w = overlap(this->x, this->w, other.x, other.w);
		float h = overlap(this->y, this->h, other.y, other.h);
		if(w < 0 || h < 0) {
			return 0;
		}
		float area = w*h;
		return area;
	}
	float union_(const Box& other) const {
		float i = intersection(other);
		float u = w*h + other.w*other.h - i;
		return u;
	}

	static float overlap(float x1, float w1, float x2, float w2)
	{
		float l1 = x1 - w1/2;
		float l2 = x2 - w2/2;
		float left = l1 > l2 ? l1 : l2;
		float r1 = x1 + w1/2;
		float r2 = x2 + w2/2;
		float right = r1 < r2 ? r1 : r2;
		return right - left;
	}
};

struct Object: yolo::Box {
	int class_id;

	friend std::istream& operator>>(std::istream& stream, Object& object) {
		return stream
			>> object.class_id
			>> object.x
			>> object.y
			>> object.w
			>> object.h;
	}
};


struct Label: std::vector<Object> {
	Label(const std::string& path) {
		std::ifstream file(path);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file " + path);
		}
		Object object;
		while (file >> object) {
			push_back(object);
		}
	}
};

struct Rect {
	int32_t left = 0, top = 0, right = 0, bottom = 0;

	float iou(const Rect& other) const {
		return float((*this & other).area()) / (*this | other).area();
	}

	int32_t area() const {
		return (right - left) * (bottom - top);
	}

	Rect operator|(const Rect& other) const {
		return {
			std::min(left, other.left),
			std::min(top, other.top),
			std::max(right, other.right),
			std::max(bottom, other.bottom)
		};
	}
	Rect& operator|=(const Rect& other) {
		left = std::min(left, other.left);
		top = std::min(top, other.top);
		right = std::max(right, other.right);
		bottom = std::max(bottom, other.bottom);
		return *this;
	}
	Rect operator&(const Rect& other) const {
		if (left > other.right) {
			return {};
		}
		if (top > other.bottom) {
			return {};
		}
		if (right < other.left) {
			return {};
		}
		if (bottom < other.top) {
			return {};
		}
		return {
			std::max(left, other.left),
			std::max(top, other.top),
			std::min(right, other.right),
			std::min(bottom, other.bottom)
		};
	}
	Rect& operator&=(const Rect& other);

	int32_t width() const {
		return right - left;
	}

	int32_t height() const {
		return bottom - top;
	}
};

struct Dataset {
	Dataset(const std::string& path)  {
		std::ifstream file(path);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file " + path);
		}
		inipp::inifile ini(file);

		item_count = std::stoi(ini.get("item_count"));

		image_width = std::stoi(ini.get("image_width"));
		image_height = std::stoi(ini.get("image_height"));
		image_channels = 3;
		image_size = image_width * image_height * image_channels;

		image_dir = ini.get("image_dir") + "/";
		label_dir = ini.get("label_dir") + "/";
	}

	std::vector<yolo::Label> charge(darknet::Network* network) {
		std::vector<yolo::Label> result;

		for (size_t row_index = 0; row_index < network->batch; ++row_index)
		{
			size_t item_index = rand() % item_count;

			std::string image_path = image_dir + std::to_string(item_index) + ".png";
			darknet::Image image(image_path);
			if ((image.w != image_width) ||
				(image.h != image_height) ||
				(image.c != image_channels)) {
				throw std::runtime_error("Image of invalid size");
			}
			memcpy(&network->input[image_size*row_index], image.data, image_size*sizeof(float));

			std::string label_path = label_dir + std::to_string(item_index) + ".txt";
			yolo::Label label(label_path);
			result.push_back(std::move(label));
		}

		return result;
	}

	size_t item_count;
	size_t image_size;
	size_t image_width, image_height, image_channels;
	std::string image_dir, label_dir;
};

bool predict(darknet::Network& network, stb::Image& image, const std::string& text);
float train(darknet::Network& network, Dataset& dataset,
	float learning_rate,
	float momentum,
	float decay);

}
