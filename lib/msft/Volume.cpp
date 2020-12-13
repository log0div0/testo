
#include "Volume.hpp"
#include <wmi/Call.hpp>

namespace msft {

Volume::Volume(wmi::WbemClassObject volume_, wmi::WbemServices services_):
	volume(std::move(volume_)), services(std::move(services_))
{

}

void Volume::format(const std::string& filesystem, const std::string& filesystemLabel) {
	try {
		auto result = services.call("Msft_Volume", "Format")
			.with("FileSystem", filesystem)
			.with("FileSystemLabel", filesystemLabel)
			.with("Full", wmi::Variant(false))
			.exec(volume);
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
