
#include "Display.hpp"

namespace hyperv {

Display::Display(wmi::WbemClassObject videoHead_, wmi::WbemClassObject virtualSystemSettingData_, wmi::WbemServices services_):
	videoHead(std::move(videoHead_)), virtualSystemSettingData(virtualSystemSettingData_), services(std::move(services_))
{
	virtualSystemManagementService = services.execQuery("SELECT * FROM Msvm_VirtualSystemManagementService").getOne();
}

std::vector<uint8_t> Display::screenshot() const {
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

size_t Display::width() const {
	return videoHead.get("CurrentHorizontalResolution").get<int32_t>();
}

size_t Display::height() const {
	return videoHead.get("CurrentVerticalResolution").get<int32_t>();
}

}
