
#include "StorageVolume.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>

namespace vir {

StorageVolume::StorageVolume(virStorageVol* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

StorageVolume::~StorageVolume() {
	if (handle) {
		virStorageVolFree(handle);
	}
}

StorageVolume::StorageVolume(StorageVolume&& other): handle(other.handle) {
	other.handle = nullptr;
}

StorageVolume& StorageVolume::operator =(StorageVolume&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string StorageVolume::name() const {
	const char* result = virStorageVolGetName(handle);
	if (!result) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	return result;
}

std::string StorageVolume::path() const {
	const char* result = virStorageVolGetPath(handle);
	if (!result) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	return result;
}

void StorageVolume::erase(std::initializer_list<virStorageVolDeleteFlags> flags) {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}
	if (virStorageVolDelete(handle, flag_bitmask) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

pugi::xml_document StorageVolume::dump_xml() const {
	char* xml = virStorageVolGetXMLDesc(handle, 0);
	if (!xml) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	pugi::xml_document result;
	result.load_string(xml);
	free(xml);
	return result;
}

}
