
#pragma once

#include "../Network.hpp"

struct QemuNetwork: Network {
	QemuNetwork() = delete;
	QemuNetwork(const QemuNetwork& other) = delete;
	QemuNetwork(const nlohmann::json& config): Network(config) {}
	~QemuNetwork() {}
};
