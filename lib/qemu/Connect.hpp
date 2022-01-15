
#pragma once

#include "XML.hpp"
#include "Domain.hpp"
#include "StoragePool.hpp"
#include "Network.hpp"
#include "Stream.hpp"
#include <vector>

namespace vir {

struct Connect {
	Connect() = default;
	Connect(virConnect* handle);
	~Connect();

	Connect(const Connect&) = delete;
	Connect& operator=(const Connect&) = delete;

	Connect(Connect&&);
	Connect& operator=(Connect&&);

	pugi::xml_document get_capabilities() const;
	pugi::xml_document get_domain_capabilities() const;

	std::vector<Domain> domains(std::initializer_list<virConnectListAllDomainsFlags> flags = {}) const;
	Domain domain_lookup_by_name(const std::string& name) const;
	Domain domain_define_xml(const pugi::xml_document& xml);
	Domain domain_define_xml_base64(const std::string& xml);

	std::vector<StoragePool> storage_pools(std::initializer_list<virConnectListAllStoragePoolsFlags> flags = {}) const;
	StoragePool storage_pool_lookup_by_name(const std::string& name) const;
	StoragePool storage_pool_define_xml(const pugi::xml_document& xml);

	StorageVolume storage_volume_lookup_by_path(const fs::path& path) const;

	std::vector<Network> networks(std::initializer_list<virConnectListAllNetworksFlags> flags = {}) const;
	Network network_lookup_by_name(const std::string& name) const;
	Network network_define_xml(const pugi::xml_document& xml);

	Stream new_stream(std::initializer_list<virStreamFlags> flags = {}); //It's not vector of flags because there's only one possible flag

	void restore_domain(const fs::path& from);

	::virConnect* handle = nullptr;
};

}
