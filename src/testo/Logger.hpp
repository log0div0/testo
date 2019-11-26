
#pragma once

#include <nlohmann/json.hpp>
#include <sstream>

struct Logger {
	Logger() = delete;
	Logger(const nlohmann::json& config);


	std::string progress() const {
		std::stringstream ss;
		ss << "[";
		ss << std::setw(3);
		ss << std::round(current_progress);
		ss << std::setw(0);
		ss << '%' << "]";
		return ss.str();
	}

	float current_progress = 0;
};

