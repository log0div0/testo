
#pragma once

#include "testo/Utils.hpp"
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

	StorageVolume volume_create_xml(const pugi::xml_node& xml, std::initializer_list<virStorageVolCreateFlags> flags = {});
	std::string dump_xml() const;
	fs::path path() const;

	::virStoragePool* handle = nullptr;
};

}
