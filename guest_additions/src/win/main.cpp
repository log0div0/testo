
#include <iostream>
#include <filesystem>
#include <locale>
#include <codecvt>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <setupapi.h>
#include <initguid.h>

namespace fs = std::filesystem;

// using convert_type = std::codecvt_utf8<wchar_t>;
// std::wstring_convert<convert_type, wchar_t> converter;

DEFINE_GUID(GUID_VIOSERIAL_PORT,
0x6fde7521, 0x1b65, 0x48ae, 0xb6, 0x28, 0x80, 0xbe, 0x62, 0x1, 0x60, 0x26);
// {6FDE7521-1B65-48ae-B628-80BE62016026}

struct HardwareDeviceInfo {
	HardwareDeviceInfo(LPGUID interfaceGuid_): interfaceGuid(interfaceGuid_) {
		handle = SetupDiGetClassDevs(
			interfaceGuid,
			NULL,
			NULL,
			(DIGCF_PRESENT | DIGCF_DEVICEINTERFACE)
		);

		if (handle == INVALID_HANDLE_VALUE) {
			throw std::runtime_error("SetupDiGetClassDevs failed");
		}
	}

	~HardwareDeviceInfo() {
		if (handle) {
			SetupDiDestroyDeviceInfoList(handle);
			handle = NULL;
		}
	}

	HardwareDeviceInfo(HardwareDeviceInfo&& other);
	HardwareDeviceInfo& operator=(HardwareDeviceInfo&& other);

	std::string getDeviceInterfaceDetail() {
		SP_DEVICE_INTERFACE_DATA deviceInterfaceData = {};
		deviceInterfaceData.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);
		BOOL result = SetupDiEnumDeviceInterfaces(handle, 0, interfaceGuid, 0, &deviceInterfaceData);
		if (result == FALSE) {
			throw std::runtime_error("SetupDiEnumDeviceInterfaces failed");
		}

		ULONG requiredLength = 0;
		SetupDiGetDeviceInterfaceDetail(handle, &deviceInterfaceData, NULL, 0, &requiredLength, NULL);
		if (requiredLength == 0) {
			throw std::runtime_error("SetupDiGetDeviceInterfaceDetail failed (1)");
		}

		std::vector<uint8_t> buffer(requiredLength, 0);
		PSP_DEVICE_INTERFACE_DETAIL_DATA deviceInterfaceDetailData = (PSP_DEVICE_INTERFACE_DETAIL_DATA)buffer.data();
		deviceInterfaceDetailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

		ULONG length = requiredLength;
		result = SetupDiGetDeviceInterfaceDetail(
			handle,
			&deviceInterfaceData,
			deviceInterfaceDetailData,
			length,
			&requiredLength,
			NULL
		);
		if (result == FALSE) {
			throw std::runtime_error("SetupDiGetDeviceInterfaceDetail failed (2)");
		}

		return deviceInterfaceDetailData->DevicePath;
	}

	LPGUID interfaceGuid = NULL;
	HDEVINFO handle = NULL;
};

int main(int argc, char** argv) {
	char szFileName[MAX_PATH];
	GetModuleFileName(NULL, szFileName, MAX_PATH);

	fs::path path(szFileName);
	path = path.replace_extension("txt");

	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path.generic_string());
	auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
	auto logger = std::make_shared<spdlog::logger>("basic_logger", spdlog::sinks_init_list{file_sink, console_sink});
	logger->set_level(spdlog::level::info);
	logger->flush_on(spdlog::level::info);
	spdlog::set_default_logger(logger);

	try {
		spdlog::info("hello world");
		HardwareDeviceInfo info((LPGUID)&GUID_VIOSERIAL_PORT);
		spdlog::info("HardwareDeviceInfo OK");
		std::string device_path = info.getDeviceInterfaceDetail();
		spdlog::info("device_path = " + device_path);
	}
	catch (const std::exception& error) {
		spdlog::error(error.what());
		return -1;
	}
}
