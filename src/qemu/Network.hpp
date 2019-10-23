
#pragma once

#include <libvirt/libvirt.h>
#include <string>

namespace vir {

struct Network {
	Network() = default;
	Network(virNetwork* handle);
	~Network();

	Network(const Network&) = delete;
	Network& operator=(const Network&) = delete;

	Network(Network&&);
	Network& operator=(Network&&);

	std::string name() const;
	bool is_active() const;

	void set_autostart(bool is_on);
	void start();

	::virNetwork* handle = nullptr;
};

}
