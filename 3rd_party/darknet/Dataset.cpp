
#include "Dataset.hpp"
#include "Image.hpp"
#include <fstream>
#include <inipp.hh>

using namespace inipp;

namespace darknet {

Data::Data(): data({}) {

}

Data::~Data() {
	free_matrix(X);
	free_matrix(y);
}

Dataset::Dataset(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	inifile ini(file);
	samples_count = std::stoi(ini.get("general", "samples_count"));
	image_width = std::stoi(ini.get("image", "width"));
	image_height = std::stoi(ini.get("image", "height"));
	image_channels = 3;
	bbox_count = std::stoi(ini.get("label", "bbox_count"));
	bbox_size = 5;
	image_dir = ini.get("image", "dir") + "/";
	label_dir = ini.get("label", "dir") + "/";
}

Dataset::~Dataset() {
}

Data Dataset::load(size_t rows_count) {
	Data d;

	d.X.vals = (float**)calloc(rows_count, sizeof(float*));
	d.X.rows = rows_count;
	d.X.cols = image_height*image_width*image_channels;

	d.y = make_matrix(rows_count, bbox_size * bbox_count);

	for (size_t row_index = 0; row_index < rows_count; ++row_index)
	{
		std::string image_path = image_dir + std::to_string(sample_index) + ".png";
		Image image(image_path);
		if ((image.w != image_width) ||
			(image.h != image_height) ||
			(image.c != image_channels)) {
			throw std::runtime_error("Image of invalid size");
		}
		std::swap(d.X.vals[row_index], image.data);

		float x, y, w, h;
		size_t bbox_class;
		size_t bbox_index = 0;
		std::string label_path = label_dir + std::to_string(sample_index) + ".txt";
		std::ifstream label(label_path);
		if (!label.is_open()) {
			throw std::runtime_error("Failed to open file " + label_path);
		}
		while (label >> bbox_class >> x >> y >> w >> h) {
			if (bbox_index == bbox_count) {
				throw std::runtime_error("Label of invalid size");
			}
			d.y.vals[row_index][bbox_size * bbox_index + 0] = x;
			d.y.vals[row_index][bbox_size * bbox_index + 1] = y;
			d.y.vals[row_index][bbox_size * bbox_index + 2] = w;
			d.y.vals[row_index][bbox_size * bbox_index + 3] = h;
			d.y.vals[row_index][bbox_size * bbox_index + 4] = bbox_class;
			++bbox_index;
		}

		sample_index = (sample_index + 1) % samples_count;
	}

	return d;
}

}
