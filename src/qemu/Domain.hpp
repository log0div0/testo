
#pragma once

#include "Snapshot.hpp"
#include "Stream.hpp"
#include "pugixml/pugixml.hpp"
#include <libvirt/libvirt.h>
#include <libvirt/libvirt-qemu.h>
#include <string>
#include <vector>

namespace vir {

struct Domain {
	Domain() = default;
	Domain(virDomain* handle);
	~Domain();

	Domain(const Domain&) = delete;
	Domain& operator=(const Domain&) = delete;

	Domain(Domain&&);
	Domain& operator=(Domain&&);

	std::string name() const;
	uint32_t id() const;
	bool is_active() const;
	virDomainState state() const;
	std::vector<Snapshot> snapshots(std::initializer_list<virDomainSnapshotListFlags> flags = {}) const;
	Snapshot snapshot_lookup_by_name(const std::string& name) const;
	Snapshot snapshot_create_xml(const pugi::xml_node& xml, std::initializer_list<virDomainSnapshotCreateFlags> flags = {});
	void revert_to_snapshot(Snapshot& snap, std::initializer_list<virDomainSnapshotRevertFlags> = {});

	pugi::xml_document dump_xml(std::initializer_list<virDomainXMLFlags> flags = {}) const;

	std::string get_metadata(virDomainMetadataType type,
		const std::string& uri,
		std::initializer_list<virDomainModificationImpact> flags = {}) const;

	void set_metadata(virDomainMetadataType type,
		const std::string& metadata,
		const std::string& key,
		const std::string& uri,
		std::vector<virDomainModificationImpact> flags = {});

	void start();
	void stop();
	void shutdown();
	void suspend();
	void resume();
	void undefine();

	operator bool() const {
		return handle != nullptr;
	}

	void send_keys(virKeycodeSet code_set, uint32_t holdtime, std::vector<uint32_t> keycodes);
	void monitor_command(const std::string& cmd, char** result, std::initializer_list<virDomainQemuMonitorCommandFlags> = {});

	void attach_device(const pugi::xml_document& xml, const std::vector<virDomainDeviceModifyFlags>& flags = {});
	void update_device(const pugi::xml_node& xml, const std::vector<virDomainDeviceModifyFlags>& flags = {});
	void detach_device(const pugi::xml_node& xml, const std::vector<virDomainDeviceModifyFlags>& flags = {});

	std::string screenshot(Stream& st, uint32_t screen_id = 0) const;
	::virDomain* handle = nullptr;
};

}
