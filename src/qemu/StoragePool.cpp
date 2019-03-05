
#include "StoragePool.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>
#include <regex>

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

std::vector<StorageVolume> StoragePool::volumes() const {
	std::vector<StorageVolume> result;

	virStorageVolPtr* volumes;

	auto size = virStoragePoolListAllVolumes(handle, &volumes, 0);
	if (size < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	for (size_t i = 0; i < size; i++) {
		result.push_back(volumes[i]);
	}

	free(volumes);
	return result;
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

StorageVolume StoragePool::volume_create_xml(const pugi::xml_node& xml, std::initializer_list<virStorageVolCreateFlags> flags) {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}
	auto res = virStorageVolCreateXML(handle, node_to_string(xml).c_str(), flag_bitmask);
	if (!res) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	return res;
}

std::string StoragePool::dump_xml() const {
	char* xml = virStoragePoolGetXMLDesc(handle, 0);
	if (!xml) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	std::string result(xml);
	free(xml);
	return result;
}

fs::path StoragePool::path() const {
	auto xml = dump_xml();

	std::regex path_regex("<path>(.*?)</path>", std::regex::ECMAScript);
	auto path_found = std::sregex_iterator(xml.begin(), xml.end(), path_regex);

	auto match = *path_found;
	std::string result(match[1].str());
	if (!result.length()) {
		throw std::runtime_error("Couldn't find attriute path in pool xml desc");
	}
	return result;
}

}
