
#pragma once

#include "Controller.hpp"
#include "Network.hpp"

struct NetworkController: public Controller {
	NetworkController();
	NetworkController(std::shared_ptr<Network> network): Controller(), network(network) {}

	std::string id() const override;
	std::string name() const override;
	std::string type() const override { return "network"; }
	bool is_defined() const override;
	void create() override;
	void create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed) override;
	void restore_snapshot(const std::string& snapshot) override;
	void delete_snapshot_with_children(const std::string& snapshot) override;

	bool has_user_key(const std::string& key);
	std::string get_user_metadata(const std::string& key);
	void set_user_metadata(const std::string& key, const std::string& value);

	bool check_config_relevance() override;

	fs::path get_metadata_dir() const override;

	std::shared_ptr<Network> network;
};
