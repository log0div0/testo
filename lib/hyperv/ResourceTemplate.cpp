
#include "ResourceTemplate.hpp"
#include <wmi/Call.hpp>

namespace hyperv {

ResourceTemplate::ResourceTemplate(wmi::WbemServices services_, const std::string& type, const std::string& subtype):
	services(std::move(services_))
{
	try {
		settings = services.execQuery(
			"SELECT * FROM " + type + " "
			"WHERE InstanceID LIKE \"%Default\" "
			"AND ResourceSubType=\"" + subtype + "\""
		).getOne().clone();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

ResourceTemplate& ResourceTemplate::put(const std::string& name, const wmi::Variant& value, CIMTYPE type) {
	try {
		settings.put(name, value, type);
		return *this;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

wmi::WbemClassObject ResourceTemplate::addTo(const wmi::WbemClassObject& systemSettings) {
	try {
		auto result = services.call("Msvm_VirtualSystemManagementService", "AddResourceSettings")
			.with("AffectedConfiguration", systemSettings.path())
			.with("ResourceSettings", std::vector<wmi::WbemClassObject>{settings})
			.exec();
		std::vector<std::string> refs = result.get("ResultingResourceSettings");
		return services.getObject(refs.at(0));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
