
#pragma once

#include "../Utils.hpp"

#include <nlohmann/json.hpp>

struct Controller {
	virtual ~Controller() {}

	virtual std::string name() const = 0;
	virtual bool is_defined() = 0;
	virtual void create() = 0;
	virtual void create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed) = 0;
	virtual void restore_snapshot(const std::string& snapshot) = 0;
	virtual void delete_snapshot_with_children(const std::string& snapshot) = 0;
	virtual bool has_snapshot(const std::string& snapshot) = 0;
	virtual std::string get_snapshot_cksum(const std::string& snapshot) = 0;

	virtual bool has_key(const std::string& key) = 0;
	virtual std::string get_metadata(const std::string& key) = 0;
	virtual void set_metadata(const std::string& key, const std::string& value) = 0;

	virtual bool check_config_relevance() = 0;

protected:
	void write_metadata_file(const fs::path& file, const nlohmann::json& metadata);
	nlohmann::json read_metadata_file(const fs::path& file) const;
};


