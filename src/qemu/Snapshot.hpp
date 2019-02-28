
#pragma once

#include "pugixml/pugixml.hpp"
#include <libvirt/libvirt.h>
#include <string>

namespace vir {

struct Snapshot {
	Snapshot() = default;
	Snapshot(virDomainSnapshot* handle);
	~Snapshot();

	Snapshot(const Snapshot&) = delete;
	Snapshot& operator=(const Snapshot&) = delete;

	Snapshot(Snapshot&&);
	Snapshot& operator=(Snapshot&&);

	std::string name() const;
	pugi::xml_document dump_xml() const;
	void destroy(std::initializer_list<virDomainSnapshotDeleteFlags> flags = {});


	::virDomainSnapshot* handle = nullptr;
};

}
