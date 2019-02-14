
#pragma once

#include "Domain.hpp"
#include <vector>

namespace qemu {

struct Connect {
	Connect() = default;
	Connect(virConnect* handle);
	~Connect();

	Connect(const Connect&) = delete;
	Connect& operator=(const Connect&) = delete;

	Connect(Connect&&);
	Connect& operator=(Connect&&);

	std::vector<Domain> ListAllDomains(virConnectListAllDomainsFlags flags) const;

	::virConnect* handle = nullptr;
};

}
