
#include "DummyNetwork.hpp"

bool DummyNetwork::is_defined() {
	return fs::exists(metadata_file());
}

