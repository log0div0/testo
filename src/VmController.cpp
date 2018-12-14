
#include <VmController.hpp>

VmController::VmController(const nlohmann::json& config):
config(config)
{
	if (!config.count("name")) {
		throw std::runtime_error("Constructing VmController error: field NAME is not specified");
	}

	if (!config.count("ram")) {
		throw std::runtime_error("Constructing VmController error: field RAM is not specified");
	}

	if (!config.count("cpus")) {
		throw std::runtime_error("Constructing VmController error: field CPUS is not specified");
	}

	if (!config.count("iso")) {
		throw std::runtime_error("Constructing VmController error: field ISO is not specified");
	}

	if (!config.count("disk_size")) {
		throw std::runtime_error("Constructing VmController error: field DISK SIZE is not specified");
	}

	if (config.count("nic")) {
		auto nics = config.at("nic");
		for (auto& nic: nics) {
			if (!nic.count("slot")) {
				throw std::runtime_error("Constructing VmController error: field slot is not specified for the nic " + 
					nic.at("name").get<std::string>());
			}
			
			if (!nic.count("attached_to")) {
				throw std::runtime_error("Constructing VmController error: field attached_to is not specified for the nic " + 
					nic.at("name").get<std::string>());
			}

			if (nic.at("attached_to").get<std::string>() == "internal") {
				if (!nic.count("network")) {
					throw std::runtime_error("Constructing VmController error: nic " + 
					nic.at("name").get<std::string>() + " has type internal, but field network is not specified");
				}
			}

			if (nic.at("attached_to").get<std::string>() == "nat") {
				if (nic.count("network")) {
					throw std::runtime_error("Constructing VmController error: nic " + 
					nic.at("name").get<std::string>() + " has type NAT, you must not specify field network");
				}
			}
		}

		for (uint32_t i = 0; i < nics.size(); i++) {
			for (uint32_t j = i + 1; j < nics.size(); j++) {
				if (nics[i].at("name") == nics[j].at("name")) {
					throw std::runtime_error("Constructing VmController error: two identical NIC names: " + 
						nics[i].at("name").get<std::string>()); 
				}

				if (nics[i].at("slot") == nics[j].at("slot")) {
					throw std::runtime_error("Constructing VmController error: two identical SLOTS: " + 
						nics[i].at("slot").get<uint32_t>()); 
				}
			}
		}
	}
}

std::string VmController::config_cksum() const {
	std::hash<std::string> h;

	auto result = h(config.dump());
	return std::to_string(result);
}

std::set<std::string> VmController::nics() const {
	std::set<std::string> result;

	if (config.count("nic")) {
		for (auto& nic: config.at("nic")) {
			result.insert(nic.at("name").get<std::string>());
		}
	}

	return result;
}

std::set<std::string> VmController::networks() const {
	std::set<std::string> result;

	if (config.count("nic")) {
		for (auto& nic: config.at("nic")) {
			auto network = nic.at("network").get<std::string>();
			if (result.find(network) == result.end()) {
				result.insert(network);
			}
		}
	}

	return result;
}
