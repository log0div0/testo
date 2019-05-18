
#include <iostream>
#include "wmi.hpp"

namespace hyperv {

struct Machine {
	Machine(wmi::WbemClassObject object_,
		wmi::WbemServices services_):
		object(std::move(object_)),
		services(std::move(services_))
	{
	}

	std::string name() const {
		try {
			return object.get("ElementName");
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	bool is_running() const {
		try {
			return object.get("EnabledState").get<int32_t>() == 2;
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	void screenshot() const {
		try {
			std::string query = "ASSOCIATORS OF {" + object.relpath() + "} WHERE ResultClass=Msvm_VideoHead";
			auto video_head = services.execQuery(query).getOne();
			int32_t height = video_head.get("CurrentVerticalResolution");
			int32_t width = video_head.get("CurrentHorizontalResolution");
			std::cout << width << " " << height << std::endl;
			auto virtualSystemManagementService = services.execQuery("SELECT * FROM Msvm_VirtualSystemManagementService").getOne();
			// services.getObject
		} catch (const std::exception&) {
			throw_with_nested(std::runtime_error(__FUNCSIG__));
		}
	}

	wmi::WbemClassObject object;
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
			machine.screenshot();
		}

	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
}
