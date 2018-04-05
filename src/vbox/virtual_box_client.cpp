
#include "virtual_box_client.hpp"
#include "error.hpp"

namespace vbox {

VirtualBoxClient::VirtualBoxClient() {
	g_pVBoxFuncs->pfnClientInitialize(NULL, &handle);
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
		VirtualBox virtual_box;
		HRESULT rc = IVirtualBoxClient_get_VirtualBox(handle, &virtual_box.handle);
		if (FAILED(rc))
		{
			throw Error(rc);
		}
		return virtual_box;
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
