
#pragma once

#include "Snapshot.hpp"
#include <libvirt/libvirt.h>
#include <string>
#include <vector>

namespace vir {

struct Domain {
	Domain() = default;
	Domain(virDomain* handle);
	~Domain();

	Domain(const Domain&) = delete;
	Domain& operator=(const Domain&) = delete;

	Domain(Domain&&);
	Domain& operator=(Domain&&);

	std::string name() const;
	bool is_active() const;


	std::vector<Snapshot> snapshots(std::initializer_list<virDomainSnapshotListFlags> flags = {}) const;
	std::string dump_xml(std::initializer_list<virDomainXMLFlags> flags = {}) const;

	void start();
	void stop();
	void undefine();


	::virDomain* handle = nullptr;
};

}
