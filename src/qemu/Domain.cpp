
#include "Domain.hpp"
#include "testo/Utils.hpp"
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

Snapshot Domain::snapshot_lookup_by_name(const std::string& name) const {
	auto result = virDomainSnapshotLookupByName(handle, name.c_str(), 0);
	if (!result) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
	return result;
}

Snapshot Domain::snapshot_create_xml(const pugi::xml_document& xml, std::initializer_list<virDomainSnapshotCreateFlags> flags) {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	auto result = virDomainSnapshotCreateXML(handle, node_to_string(xml).c_str(), flag_bitmask);
	if (!result) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	return result;
}

void Domain::revert_to_snapshot(Snapshot& snap, std::initializer_list<virDomainSnapshotRevertFlags> flags) {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	if (virDomainRevertToSnapshot(snap.handle, flag_bitmask)) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

pugi::xml_document Domain::dump_xml(std::initializer_list<virDomainXMLFlags> flags) const {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	char* xml = virDomainGetXMLDesc(handle, flag_bitmask);
	if (!xml) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	pugi::xml_document result;
	result.load_string(xml);
	free(xml);
	return result;
}

std::string Domain::get_metadata(virDomainMetadataType type,
		const std::string& uri,
		std::initializer_list<virDomainModificationImpact> flags) const
{
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	const char* uri_to_pass = uri.length() ? uri.c_str() : nullptr;

	char* metadata = virDomainGetMetadata(handle, type, uri_to_pass, flag_bitmask);
	if (!metadata) {
		throw std::runtime_error(virGetLastErrorMessage());
	}

	std::string result(metadata);
	free(metadata);
	return result;
}

void Domain::set_metadata(virDomainMetadataType type,
		const std::string& metadata,
		const std::string& key,
		const std::string& uri,
		std::vector<virDomainModificationImpact> flags)
{
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	const char* uri_to_pass = uri.length() ? uri.c_str() : nullptr;
	const char* key_to_pass = key.length() ? key.c_str() : nullptr;
	const char* metadata_to_pass = metadata.length() ? metadata.c_str() : nullptr;

	if (virDomainSetMetadata(handle, type, metadata_to_pass, key_to_pass, uri_to_pass, flag_bitmask) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

void Domain::send_keys(virKeycodeSet code_set, uint32_t holdtime, std::vector<uint32_t> keycodes) {
	if (virDomainSendKey(handle, code_set, 0, keycodes.data(), keycodes.size(), 0) < 0) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

void Domain::update_device(const pugi::xml_document& xml, std::initializer_list<virDomainDeviceModifyFlags> flags) {
	uint32_t flag_bitmask = 0;

	for (auto flag: flags) {
		flag_bitmask |= flag;
	}

	if (virDomainUpdateDeviceFlags(handle, node_to_string(xml).c_str(), flag_bitmask)) {
		throw std::runtime_error(virGetLastErrorMessage());
	}
}

}
