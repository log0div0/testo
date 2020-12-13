
#include "Bridge.hpp"
#include <wmi/Call.hpp>

namespace hyperv {

Bridge::Bridge(
	wmi::WbemClassObject virtualEthernetSwitch_,
	wmi::WbemServices services_
):
	virtualEthernetSwitch(std::move(virtualEthernetSwitch_)),
	services(std::move(services_))
{

}

std::string Bridge::name() const {
	try {
		return virtualEthernetSwitch.get("ElementName");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

void Bridge::destroy() {
	try {
		services.call("Msvm_VirtualEthernetSwitchManagementService", "DestroySystem")
			.with("AffectedSystem", virtualEthernetSwitch.path())
			.exec();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
