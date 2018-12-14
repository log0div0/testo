
#pragma once

#include "virtual_box.hpp"
#include "session.hpp"

namespace vbox {

struct VirtualBoxClient {
	VirtualBoxClient();
	~VirtualBoxClient();

	VirtualBoxClient(const VirtualBoxClient&) = delete;
	VirtualBoxClient& operator=(const VirtualBoxClient&) = delete;

	VirtualBox virtual_box() const;
	Session session() const;

	IVirtualBoxClient* handle = nullptr;
};

}
