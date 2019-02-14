
#include "Domain.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>

namespace qemu {

Domain::Domain(virDomain* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Domain::~Domain() {
	if (handle) {
		free(handle);
	}
}

Domain::Domain(Domain&& other): handle(other.handle) {
	other.handle = nullptr;
}

Domain& Domain::operator =(Domain&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Domain::name() {
	return virDomainGetName(handle);
}

}
