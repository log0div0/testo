
#include "StorageVolume.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>
#include <fstream>

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

void StorageVolume::upload_start(Stream& stream, size_t offset, size_t length, std::initializer_list<virStorageVolUploadFlags> flags) {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	if (virStorageVolUpload(handle, stream.handle, offset, length, flag_bitmask)) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

void StorageVolume::upload(Stream& stream, const std::string& file_path) {
	upload_start(stream, 0, 0);
	std::ifstream fin(file_path, std::ifstream::binary);
	if (!fin) {
		throw std::runtime_error("Can't open file to upload: " + file_path);
	}

	std::vector<uint8_t> buffer(8192);

	while(fin.read((char*)buffer.data(), buffer.size())) {
	    std::streamsize s = fin.gcount();
	    stream.send_all(buffer.data(), s);
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
