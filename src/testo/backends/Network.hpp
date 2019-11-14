
#pragma once

#include <nlohmann/json.hpp>
#include "../Utils.hpp"

struct Network {
	Network() = delete;
	Network(const nlohmann::json& config_);
	virtual ~Network() = default;

	virtual bool is_defined() = 0;
	virtual void create() = 0;
	virtual void undefine() = 0;

	std::string id() const;
	std::string name() const;
	std::string prefix() const;
	nlohmann::json get_config() const;

protected:
	nlohmann::json config;
};
