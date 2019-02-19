
#pragma once

#include "StorageVolume.hpp"
#include <libvirt/libvirt.h>
#include <string>

namespace vir {

struct StoragePool {
	StoragePool() = default;
	StoragePool(virStoragePool* handle);
	~StoragePool();

	StoragePool(const StoragePool&) = delete;
	StoragePool& operator=(const StoragePool&) = delete;

	StoragePool(StoragePool&&);
	StoragePool& operator=(StoragePool&&);

	std::string name() const;
	bool is_active() const;

	void start(std::initializer_list<virStoragePoolCreateFlags> flags = {});

	StorageVolume volume_create_xml(const std::string& xml, std::initializer_list<virStorageVolCreateFlags> flags);

	::virStoragePool* handle = nullptr;
};

}
