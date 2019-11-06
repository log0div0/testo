
#pragma once

#include "../Utils.hpp"

#include <nlohmann/json.hpp>

struct Controller {
	virtual ~Controller() {}

	virtual std::string id() const = 0;
	virtual std::string name() const = 0;
	virtual std::string type() const = 0;
	virtual bool is_defined() const;
	virtual void create() = 0;
	virtual void create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed) = 0;
	virtual void restore_snapshot(const std::string& snapshot) = 0;
	virtual void delete_snapshot_with_children(const std::string& snapshot) = 0;
	bool has_snapshot(const std::string& snapshot);
	std::string get_snapshot_cksum(const std::string& snapshot);

	bool has_key(const std::string& key);
	std::string get_metadata(const std::string& key) const;
	void set_metadata(const std::string& key, const std::string& value);

	virtual bool check_config_relevance() = 0;

	virtual fs::path get_metadata_dir() const = 0;

protected:
	fs::path main_file() const {
		fs::path result = get_metadata_dir();
		result = result / id();
		return result;
	}
};


