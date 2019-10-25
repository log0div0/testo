
#pragma once

#include "../Network.hpp"
#include <qemu/Host.hpp>

struct QemuNetwork: Network {
	QemuNetwork() = delete;
	QemuNetwork(const QemuNetwork& other) = delete;
	QemuNetwork(const nlohmann::json& config);
	~QemuNetwork() {}

	bool is_defined() override;

private:
	vir::Connect qemu_connect;
};
