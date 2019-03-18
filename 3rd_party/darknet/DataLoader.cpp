
#include "DataLoader.hpp"
#include <fstream>

namespace darknet {

Data::Data(): data({}) {

}

Data::Data(Data&& other): data(other) {
	*(data*)&other = {};
}

Data::~Data() {
	free_data(*this);
}

DataLoader::DataLoader(const std::string& path, const Trainer& trainer) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file " + path);
	}
	for (std::string line; std::getline(file, line); ) {
		paths.push_back(line);
	}
	for (auto& path: paths) {
		paths_c.push_back(path.c_str());
	}


	const Network& net = trainer.networks.back();
	layer l = net.back();

	args = get_base_args(net.impl);
	args.coords = l.coords;
	args.paths = (char**)paths_c.data();
	args.n = trainer.batch_size() * trainer.subdivisions() * trainer.networks.size();
	args.m = paths_c.size();
	args.classes = l.classes;
	args.jitter = l.jitter;
	args.num_boxes = l.max_boxes;
	args.d = &data;
	args.type = DETECTION_DATA;
	args.threads = 2;

	thread = ::load_data(args);
}

DataLoader::~DataLoader() {
	pthread_join(thread, 0);
}

Data DataLoader::load_data() {
	pthread_join(thread, 0);
	Data result = std::move(data);
	thread = ::load_data(args);
	return result;
}

}
