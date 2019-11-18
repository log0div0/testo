
#pragma once

#include "../Utils.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <set>
#include <vector>

struct FlashDrive {
	FlashDrive() = delete;
	FlashDrive(const nlohmann::json& config_);
	virtual ~FlashDrive() = default;

	virtual bool is_defined() = 0;
	virtual void create() = 0;
	virtual void undefine() = 0;
	virtual bool is_mounted() const = 0;
	virtual void mount() const = 0;
	virtual void umount() const = 0;
	virtual bool has_snapshot(const std::string& snapshot) = 0;
	virtual void make_snapshot(const std::string& snapshot) = 0;
	virtual void delete_snapshot(const std::string& snapshot) = 0;
	virtual void rollback(const std::string& snapshot) = 0;
	virtual fs::path img_path() const = 0;

	std::string id() const;
	std::string name() const;
	std::string prefix() const;
	nlohmann::json get_config() const;
	bool has_folder() const;
	void validate_folder() const;
	void load_folder() const;

protected:
	nlohmann::json config;
};
