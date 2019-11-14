
#pragma once

#include "Network.hpp"

struct NetworkController {
	NetworkController() = delete;
	NetworkController(std::shared_ptr<Network> network): network(network) {}
	NetworkController(const fs::path& main_file); //used only for cleanup purposes

	std::string id() const;
	std::string name() const;
	std::string prefix() const;
	bool is_defined() const;
	void create();
	void undefine();

	bool check_config_relevance();

	fs::path get_metadata_dir() const;
	std::string get_metadata(const std::string& key) const;

	std::shared_ptr<Network> network;

private:
	fs::path main_file() const {
		fs::path result = get_metadata_dir();
		result = result / id();
		return result;
	}
};
