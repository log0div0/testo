
#pragma once

#include <libvirt/libvirt.h>
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

	void erase(std::initializer_list<virStorageVolDeleteFlags> flags = {});

	::virStorageVol* handle = nullptr;
};

}
