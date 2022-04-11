
#pragma once

#include "Controller.hpp"
#include "../backends/Network.hpp"

namespace IR {

struct Network: Controller {
	static std::string type_name() { return "network"; }

	std::shared_ptr<::Network> nw() const;

	virtual std::string type() const override;

	virtual void create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed) override;
	virtual void restore_snapshot(const std::string& snapshot) override;
	virtual bool has_hypervisor_snapshot(const std::string& snapshot) override;
	virtual void delete_hypervisor_snapshot(const std::string& snapshot) override;
	virtual void delete_snapshot_with_children(const std::string& snapshot) override;

	virtual bool check_config_relevance() override;

	virtual bool is_defined() const override;
	virtual void create() override;
	virtual void undefine() override;

	void validate_config();

private:
	virtual std::string id() const override;
	virtual fs::path get_metadata_dir() const override;

	mutable std::shared_ptr<::Network> _nw;
};

}
