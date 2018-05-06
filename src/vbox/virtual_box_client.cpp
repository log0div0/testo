
#include "virtual_box_client.hpp"
#include "throw_if_failed.hpp"

namespace vbox {

VirtualBoxClient::VirtualBoxClient() {
	try {
		throw_if_failed(api->pfnClientInitialize(NULL, &handle));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

VirtualBoxClient::~VirtualBoxClient() {
	if (handle) {
		IVirtualBoxClient_Release(handle);
	}
}

VirtualBox VirtualBoxClient::virtual_box() const {
	try {
		IVirtualBox* result = nullptr;
		throw_if_failed(IVirtualBoxClient_get_VirtualBox(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Session VirtualBoxClient::session() const {
	try {
		ISession* result = nullptr;
		throw_if_failed(IVirtualBoxClient_get_Session(handle, &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
