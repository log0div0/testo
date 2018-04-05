
#pragma once

#include <VBoxCAPIGlue/VBoxCAPIGlue.h>
#include "virtual_box.hpp"

namespace vbox {

struct VirtualBoxClient {
	VirtualBoxClient();
	~VirtualBoxClient();

	VirtualBoxClient(const VirtualBoxClient&) = delete;
	VirtualBoxClient& operator=(const VirtualBoxClient&) = delete;

	VirtualBox virtual_box() const;

	IVirtualBoxClient* handle = nullptr;
};

}
