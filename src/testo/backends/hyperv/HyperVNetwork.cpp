
#include "HyperVNetwork.hpp"

HyperVNetwork::HyperVNetwork(const nlohmann::json& config): Network(config) {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

bool HyperVNetwork::is_defined() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void HyperVNetwork::create() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void HyperVNetwork::undefine() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
