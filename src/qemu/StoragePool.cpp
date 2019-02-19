
#include "StoragePool.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>

namespace vir {

StoragePool::StoragePool(virStoragePool* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

StoragePool::~StoragePool() {
	if (handle) {
		virStoragePoolFree(handle);
	}
}

StoragePool::StoragePool(StoragePool&& other): handle(other.handle) {
	other.handle = nullptr;
}

StoragePool& StoragePool::operator =(StoragePool&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string StoragePool::name() const {
	return virStoragePoolGetName(handle);
}

bool StoragePool::is_active() const {
	auto res = virStoragePoolIsActive(handle);
	if (res < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return res;
}

void StoragePool::start(std::initializer_list<virStoragePoolCreateFlags> flags) {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}
	auto res = virStoragePoolCreate(handle, flag_bitmask);
	if (res < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

StorageVolume StoragePool::volume_create_xml(const std::string& xml, std::initializer_list<virStorageVolCreateFlags> flags) {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}
	auto res = virStorageVolCreateXML(handle, xml.c_str(), flag_bitmask);
	if (!res) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	return res;
}

}
