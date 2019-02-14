
#pragma once

#include <libvirt/libvirt.h>
#include <string>

namespace qemu {

struct Domain {
	Domain() = default;
	Domain(virDomain* handle);
	~Domain();

	Domain(const Domain&) = delete;
	Domain& operator=(const Domain&) = delete;

	Domain(Domain&&);
	Domain& operator=(Domain&&);

	std::string name();

	::virDomain* handle = nullptr;
};

}
