
#pragma once

#include "api.hpp"

namespace vbox {

struct VirtualBoxErrorInfo {
	VirtualBoxErrorInfo() = default;
	VirtualBoxErrorInfo(IVirtualBoxErrorInfo* handle);
	~VirtualBoxErrorInfo();

	std::string text() const;
	operator bool() const;

	VirtualBoxErrorInfo(const VirtualBoxErrorInfo&) = delete;
	VirtualBoxErrorInfo& operator=(const VirtualBoxErrorInfo&) = delete;
	VirtualBoxErrorInfo(VirtualBoxErrorInfo&& other);
	VirtualBoxErrorInfo& operator=(VirtualBoxErrorInfo&& other);

	IVirtualBoxErrorInfo* handle = nullptr;
};

}
