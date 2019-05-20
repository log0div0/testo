
#include "Machine.hpp"

namespace hyperv {

Machine::Machine(wmi::WbemClassObject computerSystem_,
	wmi::WbemServices services_):
	computerSystem(std::move(computerSystem_)),
	services(std::move(services_))
{
	try {
		virtualSystemSettingData = services.execQuery("ASSOCIATORS OF {" + computerSystem.relpath() + "} WHERE ResultClass=Msvm_VirtualSystemSettingData").getOne();
		virtualSystemManagementService = services.execQuery("SELECT * FROM Msvm_VirtualSystemManagementService").getOne();
		videoHead = services.execQuery("ASSOCIATORS OF {" + computerSystem.relpath() + "} WHERE ResultClass=Msvm_VideoHead").getOne();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::string Machine::name() const {
	try {
		return computerSystem.get("ElementName");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

bool Machine::is_running() const {
	try {
		return computerSystem.get("EnabledState").get<int32_t>() == 2;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<uint8_t> Machine::screenshot() const {
	try {
		auto call = services.getObject("Msvm_VirtualSystemManagementService").getMethod("GetVirtualSystemThumbnailImage").spawnInstance();
		call.put("HeightPixels", videoHead.get("CurrentVerticalResolution"));
		call.put("WidthPixels", videoHead.get("CurrentHorizontalResolution"));
		call.put("TargetSystem", virtualSystemSettingData.path());
		auto result = services.execMethod(virtualSystemManagementService.path(), "GetVirtualSystemThumbnailImage", call);
		if (result.get("ReturnValue").get<int32_t>() != 0) {
			throw std::runtime_error("ReturnValue == " + std::to_string(result.get("ReturnValue").get<int32_t>()));
		}

		return result.get("ImageData");
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

size_t Machine::screenWidth() const {
	return videoHead.get("CurrentHorizontalResolution").get<int32_t>();
}

size_t Machine::screenHeight() const {
	return videoHead.get("CurrentVerticalResolution").get<int32_t>();
}

}
