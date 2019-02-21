
#include "Connect.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>

namespace vir {

Connect::Connect(virConnect* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Connect::~Connect() {
	if (handle) {
		virConnectClose(handle);
	}
}

Connect::Connect(Connect&& other): handle(other.handle) {
	other.handle = nullptr;
}

Connect& Connect::operator =(Connect&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::vector<Domain> Connect::domains(std::initializer_list<virConnectListAllDomainsFlags> flags) const {
	std::vector<Domain> result;

	virDomainPtr* domains;

	uint32_t flags_bimask = 0;
	for (auto flag: flags) {
		flags_bimask |= flag;
	}

	auto size = virConnectListAllDomains(handle, &domains, flags_bimask);
	if (size < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	for (size_t i = 0; i < size; i++) {
		result.push_back(domains[i]);
	}

	free(domains);
	return result;
}

Domain Connect::domain_lookup_by_name(const std::string& name) const {
	auto result = virDomainLookupByName(handle, name.c_str());
	if (!result) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	return result;
}

Domain Connect::domain_define_xml(const std::string& xml) {
	auto result = virDomainDefineXML(handle, xml.c_str());
	if (!result) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return result;
}

StorageVolume Connect::storage_volume_lookup_by_path(const fs::path& path) const {
	auto result = virStorageVolLookupByPath(handle, path.generic_string().c_str());
	if (!result) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return result;

}

std::vector<StoragePool> Connect::storage_pools(std::initializer_list<virConnectListAllStoragePoolsFlags> flags) const {
	std::vector<StoragePool> result;

	virStoragePoolPtr* pools;

	uint32_t flags_bimask = 0;
	for (auto flag: flags) {
		flags_bimask |= flag;
	}

	auto size = virConnectListAllStoragePools(handle, &pools, flags_bimask);
	if (size < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	for (size_t i = 0; i < size; i++) {
		result.push_back(pools[i]);
	}

	free(pools);
	return result;
}

StoragePool Connect::storage_pool_lookup_by_name(const std::string& name) const {
	auto res = virStoragePoolLookupByName(handle, name.c_str());
	if (!res) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return res;
}

StoragePool Connect::storage_pool_define_xml(const std::string& xml) {
	auto result = virStoragePoolDefineXML(handle, xml.c_str(), 0);
	if (!result) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return result;
}

std::vector<Network> Connect::networks(std::initializer_list<virConnectListAllNetworksFlags> flags) const {
	std::vector<Network> result;

	virNetworkPtr* nets;

	uint32_t flags_bimask = 0;
	for (auto flag: flags) {
		flags_bimask |= flag;
	}

	auto size = virConnectListAllNetworks(handle, &nets, flags_bimask);
	if (size < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	for (size_t i = 0; i < size; i++) {
		result.push_back(nets[i]);
	}

	free(nets);
	return result;
}

Network Connect::network_define_xml(const std::string& xml) {
	auto result = virNetworkDefineXML(handle, xml.c_str());
	if (!result) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return result;
}

}
