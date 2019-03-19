
#pragma once

#include "include/darknet.h"
#include <string>
#include <vector>

namespace darknet {

struct Data: data {
	Data();
	~Data();

	Data(const Data&) = delete;
	Data& operator=(const Data&) = delete;

	Data(Data&&);
	Data& operator=(Data&&);
};

struct Dataset {
	Dataset(const std::string& path);
	~Dataset();

	Dataset(const Dataset&) = delete;
	Dataset& operator=(const Dataset&) = delete;

	Dataset(Dataset&&);
	Dataset& operator=(Dataset&&);

	Data load(size_t size);

private:
	size_t sample_index = 0;
	size_t samples_count;
	size_t image_width, image_height, image_channels, bbox_size, bbox_count;
	std::string image_dir, label_dir;
};

}
