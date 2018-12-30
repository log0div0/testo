
#include <vbox/network_adapter.hpp>
#include <vbox/throw_if_failed.hpp>
#include <vbox/string.hpp>

namespace vbox {

NetworkAdapter::NetworkAdapter(INetworkAdapter* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

NetworkAdapter::~NetworkAdapter() {
	if (handle) {
		INetworkAdapter_Release(handle);
	}
}

NetworkAdapter::NetworkAdapter(NetworkAdapter&& other): handle(other.handle) {
	other.handle = nullptr;
}

NetworkAdapter& NetworkAdapter::operator=(NetworkAdapter&& other) {
	std::swap(handle, other.handle);
	return *this;
}

void NetworkAdapter::setCableConnected(bool is_connected) {
	try {
		throw_if_failed(INetworkAdapter_put_CableConnected(handle, is_connected));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void NetworkAdapter::setEnabled(bool is_enabled) const {
	try {
		throw_if_failed(INetworkAdapter_put_Enabled(handle, is_enabled));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void NetworkAdapter::setInternalNetwork(const std::string& network) const {
	try {
		throw_if_failed(INetworkAdapter_put_InternalNetwork(handle, StringIn(network)));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void NetworkAdapter::setAttachmentType(NetworkAttachmentType type) const {
	try {
		throw_if_failed(INetworkAdapter_put_AttachmentType(handle, type));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}


void NetworkAdapter::setAdapterType(NetworkAdapterType type) const {
	try {
		throw_if_failed(INetworkAdapter_put_AdapterType(handle, type));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void NetworkAdapter::setMAC(const std::string& mac) {
	try {
		throw_if_failed(INetworkAdapter_put_MACAddress(handle, StringIn(mac)));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
