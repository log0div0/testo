
#include "Connect.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>

namespace qemu {

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

std::vector<Domain> Connect::ListAllDomains(virConnectListAllDomainsFlags flags) const {
	std::vector<Domain> result;

	virDomainPtr* domains;

	auto size = virConnectListAllDomains(handle, &domains, flags);
	if (size < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	for (size_t i = 0; i < size; i++) {
		result.push_back(domains[i]);
	}

	free(domains);

	return result;

}


}
