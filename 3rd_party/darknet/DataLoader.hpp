
#pragma once

#include "Trainer.hpp"

namespace darknet {

struct Data: data {
	Data();
	~Data();

	Data(const Data&) = delete;
	Data& operator=(const Data&) = delete;

	Data(Data&&);
	Data& operator=(Data&&);
};

struct DataLoader {
	DataLoader(const std::string& path, const Trainer& trainer);
	~DataLoader();

	DataLoader(const DataLoader&) = delete;
	DataLoader& operator=(const DataLoader&) = delete;

	DataLoader(DataLoader&&);
	DataLoader& operator=(DataLoader&&);

	Data load_data();

private:
	std::vector<std::string> paths;
	std::vector<const char*> paths_c;
	load_args args;
	Data data;
	pthread_t thread;
};

}
