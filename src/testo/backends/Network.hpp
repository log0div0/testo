
#pragma once

#include <nlohmann/json.hpp>
#include "../Utils.hpp"

struct Network {
	Network() = delete;
	Network(const nlohmann::json& config_);
	virtual ~Network() = default;

	virtual bool is_defined() = 0;

	std::string id() const;
	std::string name() const;
	nlohmann::json get_config() const;

protected:
	nlohmann::json config;
};
