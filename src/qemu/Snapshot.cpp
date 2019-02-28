
#include "Snapshot.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>

namespace vir {

Snapshot::Snapshot(virDomainSnapshot* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Snapshot::~Snapshot() {
	if (handle) {
		virDomainSnapshotFree(handle);
	}
}

Snapshot::Snapshot(Snapshot&& other): handle(other.handle) {
	other.handle = nullptr;
}

Snapshot& Snapshot::operator =(Snapshot&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Snapshot::name() const {
	return virDomainSnapshotGetName(handle);
}

pugi::xml_document Snapshot::dump_xml() const {
	char* xml = virDomainSnapshotGetXMLDesc(handle, 0);
	if (!xml) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	pugi::xml_document result;
	result.load_string(xml);
	free(xml);
	return result;
}


void Snapshot::destroy(std::initializer_list<virDomainSnapshotDeleteFlags> flags) {
	uint32_t flags_bitmask = 0;
	for (auto flag: flags) {
		flags_bitmask |= flag;
	}
	if (virDomainSnapshotDelete(handle, flags_bitmask) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

}
