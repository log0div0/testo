
#include "Snapshot.hpp"
#include <wmi/Call.hpp>

namespace hyperv {

Snapshot::Snapshot(wmi::WbemClassObject virtualSystemSettingData_,
	wmi::WbemServices services_):
	virtualSystemSettingData(std::move(virtualSystemSettingData_)),
	services(std::move(services_))
{

}

std::string Snapshot::name() const {
	try {
		return virtualSystemSettingData.get("ElementName");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Snapshot::setName(const std::string& name) {
	try {
		virtualSystemSettingData.put("ElementName", name);
		services.call("Msvm_VirtualSystemManagementService", "ModifySystemSettings")
			.with("SystemSettings", virtualSystemSettingData)
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Snapshot::apply() {
	try {
		services.call("Msvm_VirtualSystemSnapshotService", "ApplySnapshot")
			.with("Snapshot", virtualSystemSettingData.path())
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Snapshot::destroy() {
	try {
		services.call("Msvm_VirtualSystemSnapshotService", "DestroySnapshot")
			.with("AffectedSnapshot", virtualSystemSettingData.path())
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
