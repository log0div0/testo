
#pragma once

#include "../Utils.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <set>
#include <vector>

struct FlashDriveController {
	FlashDriveController() = delete;
	FlashDriveController(const nlohmann::json& config_);
	virtual ~FlashDriveController() = default;

	virtual void create() = 0;
	virtual bool is_mounted() const = 0;
	virtual void mount() const = 0;
	virtual void umount() const = 0;
	virtual void load_folder() const = 0;
	virtual fs::path img_path() const = 0;
	std::string cksum() const;
	bool is_cksum_ok() const {
		return calc_cksum() == read_cksum();
	}

	std::string name() const;
	nlohmann::json get_config() const;
	bool has_folder() const;
	bool cache_enabled() const;

protected:
	std::string read_cksum() const;
	void write_cksum(const std::string& cksum) const;
	std::string calc_cksum() const;
	void delete_cksum() const;
	fs::path cksum_path() const;
	nlohmann::json config;
};
