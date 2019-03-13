
#include "Network.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>

namespace vir {

Network::Network(virNetwork* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Network::~Network() {
	if (handle) {
		virNetworkFree(handle);
	}
}

Network::Network(Network&& other): handle(other.handle) {
	other.handle = nullptr;
}

Network& Network::operator =(Network&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Network::name() const {
	return virNetworkGetName(handle);
}

bool Network::is_active() const {
	auto res = virNetworkIsActive(handle);
	if (res < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return res;
}

void Network::start() {
	if (virNetworkCreate(handle) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

}
