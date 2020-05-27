
#pragma once

#include "XML.hpp"
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
	StorageVolume storage_volume_lookup_by_name(const std::string& name) const;
	std::vector<StorageVolume> volumes() const;

	void start(std::initializer_list<virStoragePoolCreateFlags> flags = {});
	void refresh();

	StorageVolume volume_create_xml(const pugi::xml_node& xml, std::initializer_list<virStorageVolCreateFlags> flags = {});
	std::string dump_xml() const;
	fs::path path() const;

	::virStoragePool* handle = nullptr;
};

}
