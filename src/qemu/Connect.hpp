
#pragma once

#include "testo/Utils.hpp"
#include "Domain.hpp"
#include "StoragePool.hpp"
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

	std::vector<Domain> domains(std::initializer_list<virConnectListAllDomainsFlags> flags = {}) const;
	Domain domain_lookup_by_name(const std::string& name) const;
	StorageVolume storage_volume_lookup_by_path(const fs::path& path) const;


	std::vector<StoragePool> storage_pools(std::initializer_list<virConnectListAllStoragePoolsFlags> flags = {}) const;
	StoragePool storage_pool_define_xml(const std::string& xml);


	::virConnect* handle = nullptr;
};

}
