
#include "virtual_box_client.hpp"
#include "error.hpp"

namespace vbox {

VirtualBoxClient::VirtualBoxClient() {
	api->pfnClientInitialize(NULL, &handle);
	if (!handle)
	{
		throw std::runtime_error(__PRETTY_FUNCTION__);
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
		HRESULT rc = IVirtualBoxClient_get_VirtualBox(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Session VirtualBoxClient::session() const {
	try {
		ISession* result = nullptr;
		HRESULT rc = IVirtualBoxClient_get_Session(handle, &result);
		if (FAILED(rc)) {
			throw Error(rc);
		}
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
