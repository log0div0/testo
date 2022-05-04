
#pragma once

#include "Object.hpp"

#define TESTO_CURRENT_METADATA_VERSION 3

namespace IR {

struct Controller: Object<AST::Controller> {
	virtual ~Controller() {}

	virtual std::string type() const = 0;

	std::string name() const;

	std::string note_was_declared_here() const;

	std::string get_snapshot_cksum(const std::string& snapshot);
	bool has_snapshot(const std::string& snapshot, bool hypervisor_snapshot_needed = false);
	bool check_metadata_version();
	virtual void create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed) = 0;
	virtual void restore_snapshot(const std::string& snapshot) = 0;
	virtual bool has_hypervisor_snapshot(const std::string& snapshot) = 0;
	virtual void delete_hypervisor_snapshot(const std::string& snapshot) = 0;
	virtual void delete_snapshot_with_children(const std::string& snapshot) = 0;

	//To be called only if the
	virtual bool check_config_relevance() = 0;

	nlohmann::json get_metadata(const std::string& key) const;
	void set_metadata(const std::string& key, const nlohmann::json& value);

	static nlohmann::json get_metadata(const fs::path& file, const std::string& key);
	static void set_metadata(const fs::path& file, const std::string& key, const nlohmann::json& value);

	static nlohmann::json read_metadata_file(const fs::path& file);
	static void write_metadata_file(const fs::path& file, const nlohmann::json& metadata);

	virtual bool is_defined() const;
	virtual void create() = 0;
	virtual void undefine() = 0;

	nlohmann::json config;
	std::string current_state;

protected:
	virtual std::string id() const = 0;
	virtual fs::path get_metadata_dir() const = 0;
	fs::path main_file() const;
	bool has_key(const std::string& key);
};

}
