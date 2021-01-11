
#include "HyperVNetwork.hpp"

HyperVNetwork::HyperVNetwork(const nlohmann::json& config): Network(config) {
}

bool HyperVNetwork::is_defined() {
	auto bridges = connect.bridges();
	auto it = std::find_if(bridges.begin(), bridges.end(), [&](auto bridge) {
		return bridge.name() == id();
	});
	return it != bridges.end();
}

void HyperVNetwork::create() {
	hyperv::Bridge bridge = connect.defineBridge(id());
}

void HyperVNetwork::undefine() {
	hyperv::Bridge bridge = connect.bridge(id());
	bridge.destroy();
}
