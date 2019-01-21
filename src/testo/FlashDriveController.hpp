
#pragma once

#include "Utils.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <set>
#include <vector>

struct FlashDriveController {
	virtual ~FlashDriveController() = default;
	virtual int create() = 0;
	virtual bool is_mounted() const = 0;
	virtual int mount() const = 0;
	virtual int umount() const = 0;
	virtual int load_folder() const = 0;

	virtual std::string name() const = 0;
	virtual fs::path img_path() const = 0;
	virtual bool has_folder() const = 0;
};
