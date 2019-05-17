
#pragma once

#include "Utils.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <set>
#include <vector>

struct FlashDriveController {
	virtual ~FlashDriveController() = default;
	virtual void create() = 0;
	virtual bool is_mounted() const = 0;
	virtual void mount() const = 0;
	virtual void umount() const = 0;
	virtual void load_folder() const = 0;

	virtual fs::path img_path() const = 0;
	virtual std::string name() const = 0;
	virtual nlohmann::json get_config() const = 0;
	virtual bool has_folder() const = 0;
	virtual std::string cksum() const = 0;
	virtual bool cache_enabled() const = 0;
};
