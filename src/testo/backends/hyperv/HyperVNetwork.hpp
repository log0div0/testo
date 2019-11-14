
#pragma once

#include "../Network.hpp"

struct HyperVNetwork: Network {
	HyperVNetwork() = delete;
	HyperVNetwork(const HyperVNetwork& other) = delete;
	HyperVNetwork(const nlohmann::json& config);
	~HyperVNetwork() {}

	bool is_defined() override;
	void create() override;
	void undefine() override;
};
