
#include "Display.hpp"

namespace hyperv {

Display::Display(wmi::WbemClassObject videoHead_, wmi::WbemServices services_):
	videoHead(std::move(videoHead_)), services(std::move(services_))
{
}

std::vector<uint8_t> Display::screenshot() const {
	try {
		auto virtualSystemSettingData = services.execQuery("SELECT * FROM Msvm_VirtualSystemSettingData WHERE InstanceID=\"Microsoft:" + videoHead.get("SystemName").get<std::string>() + "\"").getOne();
		return services.call("Msvm_VirtualSystemManagementService", "GetVirtualSystemThumbnailImage")
			.with("HeightPixels", videoHead.get("CurrentVerticalResolution"))
			.with("WidthPixels", videoHead.get("CurrentHorizontalResolution"))
			.with("TargetSystem", virtualSystemSettingData.path())
			.exec()
			.get("ImageData");
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
