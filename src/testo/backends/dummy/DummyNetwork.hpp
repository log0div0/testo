
#pragma once

#include "../Network.hpp"

struct DummyNetwork: Network {
	DummyNetwork() = delete;
	DummyNetwork(const DummyNetwork& other) = delete;
	DummyNetwork(const nlohmann::json& config): Network(config) {}
	~DummyNetwork() {}
};
