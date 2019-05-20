
#include "Connect.hpp"

namespace hyperv {

Connect::Connect() {
	try {
		services = locator.connectServer("root\\virtualization\\v2");
		services.setProxyBlanket();
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

std::vector<Machine> Connect::machines() const {
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

}
