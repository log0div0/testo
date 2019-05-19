
#include <iostream>
#include <chrono>
#include "wmi.hpp"

namespace hyperv {

struct Machine {
	Machine(wmi::WbemClassObject computerSystem_,
		wmi::WbemServices services_):
		computerSystem(std::move(computerSystem_)),
		services(std::move(services_))
	{
		videoHead = services.execQuery("ASSOCIATORS OF {" + computerSystem.relpath() + "} WHERE ResultClass=Msvm_VideoHead").getOne();
		virtualSystemSettingData = services.execQuery("ASSOCIATORS OF {" + computerSystem.relpath() + "} WHERE ResultClass=Msvm_VirtualSystemSettingData").getOne();
		virtualSystemManagementService = services.execQuery("SELECT * FROM Msvm_VirtualSystemManagementService").getOne();
	}

	std::string name() const {
		try {
			return computerSystem.get("ElementName");
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	bool is_running() const {
		try {
			return computerSystem.get("EnabledState").get<int32_t>() == 2;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	std::vector<uint8_t> screenshot() const {
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

	wmi::WbemClassObject computerSystem, videoHead, virtualSystemSettingData, virtualSystemManagementService;
	wmi::WbemServices services;
};

struct Connect {
	Connect() {
		try {
			services = locator.connectServer("root\\virtualization\\v2");
			services.setProxyBlanket();
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	std::vector<Machine> machines() const {
		try {
			std::vector<Machine> result;
			auto objects = services.execQuery("SELECT * FROM Msvm_ComputerSystem WHERE Caption=\"Virtual Machine\"").getAll();
			for (auto& object: objects) {
				result.push_back(Machine(std::move(object), services));
			}
			return result;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	wmi::WbemLocator locator;
	wmi::WbemServices services;
};

}

void backtrace(std::ostream& stream, const std::exception& error, size_t n) {
	stream << n << ". " << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << std::endl;
		backtrace(stream, error, n + 1);
	} catch(...) {
		stream << std::endl;
		stream << n << ". " << "[Unknown exception type]";
	}
}

std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error, 1);
	return stream;
}

void main() {
	try {
		wmi::CoInitializer initializer;
		initializer.initalize_security();

		hyperv::Connect connect;
		for (auto& machine: connect.machines()) {
			std::cout << machine.name() << " " << (machine.is_running() ? "running" : "stopped") << std::endl;
			auto start = std::chrono::high_resolution_clock::now();
			std::vector<uint8_t> screenshot = machine.screenshot();
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time = end - start;
			std::cout << time.count() << " seconds" << std::endl;
			std::cout << "SIZE = " << screenshot.size() << std::endl;
		}

	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
}
