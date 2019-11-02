
#pragma once

#include <libvirt/libvirt.h>
#include "pugixml/pugixml.hpp"
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
	bool is_persistent() const;

	pugi::xml_document dump_xml(std::initializer_list<virNetworkXMLFlags> flags = {}) const;

	void set_autostart(bool is_on);
	void start();
	void stop();

	void undefine();

	::virNetwork* handle = nullptr;
};

}
