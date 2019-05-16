
#include "wmi.hpp"
#include <iostream>

namespace hyperv {

struct Machine {
	Machine(wmi::WbemClassObject object_): object(std::move(object_)) {

	}

	std::string name() const {
		return object.get<std::string>("ElementName");
	}

private:
	wmi::WbemClassObject object;
};

struct Connect {
	Connect() {
		services = locator.connectServer("root\\virtualization\\v2");
		services.setProxyBlanket();
	}

	std::vector<Machine> machines() const {
		auto objects = services.execQuery("SELECT * FROM Msvm_ComputerSystem WHERE Caption = \"Virtual Machine\"").getAll();
		return {std::make_move_iterator(objects.begin()), std::make_move_iterator(objects.end())};
	}

private:
	wmi::WbemLocator locator;
	wmi::WbemServices services;
};

}

void main() {
	try {
		wmi::CoInitializer initializer;
		initializer.initalize_security();

		hyperv::Connect connect;
		for (auto& machine: connect.machines()) {
			std::cout << machine.name() << std::endl;
		}

	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
	}
}
