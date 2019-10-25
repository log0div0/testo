
#pragma once

#include "../Network.hpp"

struct VboxNetwork: Network {
	VboxNetwork() = delete;
	VboxNetwork(const VboxNetwork& other) = delete;
	VboxNetwork(const nlohmann::json& config): Network(config) {}
	~VboxNetwork() {}

	bool is_defined() override;
};
