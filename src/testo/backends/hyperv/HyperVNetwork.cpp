
#include "HyperVNetwork.hpp"

HyperVNetwork::HyperVNetwork(const nlohmann::json& config): Network(config) {
	std::cout << "HyperVNetwork " << config.dump(4) << std::endl;
}

bool HyperVNetwork::is_defined() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}

void HyperVNetwork::create() {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
}

void HyperVNetwork::undefine() {
	std::cout << "TODO: " << __PRETTY_FUNCTION__ << std::endl;
}
