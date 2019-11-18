
#include "DummyNetwork.hpp"

bool DummyNetwork::is_defined() {
	return fs::exists(metadata_file());
}

void DummyNetwork::create() {
	//remove the file if it exists

	if (fs::exists(metadata_file())) {
		fs::remove(metadata_file());
	}

	nlohmann::json config;

	write_metadata_file(metadata_file(), config);
}

void DummyNetwork::undefine() {
	//remove the file if it exists

	if (fs::exists(metadata_file())) {
		fs::remove(metadata_file());
	}
}
