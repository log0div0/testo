
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

bool Network::is_persistent() const {
	auto res = virNetworkIsPersistent(handle);
	if (res < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return res;
}


pugi::xml_document Network::dump_xml(std::initializer_list<virNetworkXMLFlags> flags) const {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	char* xml = virNetworkGetXMLDesc(handle, flag_bitmask);
	if (!xml) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	pugi::xml_document result;
	result.load_string(xml);
	free(xml);
	return result;
}

void Network::set_autostart(bool is_on) {
	if(virNetworkSetAutostart(handle, (int)is_on) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

}

void Network::start() {
	if (virNetworkCreate(handle) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

void Network::stop() {
	if (virNetworkDestroy(handle) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

void Network::undefine() {
	if (virNetworkUndefine(handle) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}
}
