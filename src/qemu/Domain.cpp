
#include "Domain.hpp"
#include <libvirt/virterror.h>
#include <stdexcept>
namespace vir {

Domain::Domain(virDomain* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

Domain::~Domain() {
	if (handle) {
		virDomainFree(handle);
	}
}

Domain::Domain(Domain&& other): handle(other.handle) {
	other.handle = nullptr;
}

Domain& Domain::operator =(Domain&& other) {
	std::swap(handle, other.handle);
	return *this;
}

std::string Domain::name() const {
	return virDomainGetName(handle);
}

bool Domain::is_active() const {
	int result = virDomainIsActive(handle);
	if (result < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return result;
}

void Domain::start() {
	if (virDomainCreate(handle) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

void Domain::stop() {
	if (virDomainDestroy(handle) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

void Domain::undefine() {
	if (virDomainUndefine(handle) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

std::vector<Snapshot> Domain::snapshots(std::initializer_list<virDomainSnapshotListFlags> flags) const {
	std::vector<Snapshot> result;

	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	virDomainSnapshotPtr* snapshots;

	auto size = virDomainListAllSnapshots(handle, &snapshots, flag_bitmask);
	if (size < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	for (size_t i = 0; i < size; i++) {
		result.push_back(snapshots[i]);
	}

	free(snapshots);
	return result;
}

std::string Domain::dump_xml(std::initializer_list<virDomainXMLFlags> flags) const {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	char* xml = virDomainGetXMLDesc(handle, flag_bitmask);
	if (!xml) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	std::string result(xml);
	free(xml);
	return result;
}

void Domain::send_keys(virKeycodeSet code_set, uint32_t holdtime, std::vector<uint32_t> keycodes) {
	uint32_t enter = 0x28;
	if (virDomainSendKey(handle, code_set, 0, keycodes.data(), keycodes.size(), 0) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

}
