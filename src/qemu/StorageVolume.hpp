
#pragma once

#include <libvirt/libvirt.h>
#include "pugixml/pugixml.hpp"
#include <string>
#include <vector>

namespace vir {

struct StorageVolume {
	StorageVolume() = default;
	StorageVolume(virStorageVol* handle);
	~StorageVolume();

	StorageVolume(const StorageVolume&) = delete;
	StorageVolume& operator=(const StorageVolume&) = delete;

	StorageVolume(StorageVolume&&);
	StorageVolume& operator=(StorageVolume&&);

	std::string name() const;
	std::string path() const;

	void erase(std::initializer_list<virStorageVolDeleteFlags> flags = {});

	pugi::xml_document dump_xml() const;

	::virStorageVol* handle = nullptr;
};

}
