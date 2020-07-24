
#pragma once

#include <string>
#include <regex>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

namespace license {

struct Date {
	Date(const std::string& str) {
		std::regex regex(R"((\d+).(\d+).(\d+))");
		std::cmatch match;
		if (!std::regex_match(str.data(), match, regex)) {
			throw std::runtime_error("Invalid date format");
		}
		day = stoi(match[1]);
		month = stoi(match[2]);
		year = stoi(match[3]);

		if (month > 12) {
			throw std::runtime_error("Invalid month number");
		}
		if (day > 31) {
			throw std::runtime_error("Invalid day number");
		}
	}

	Date(const std::chrono::system_clock::time_point& time_point) {
		std::time_t time = std::chrono::system_clock::to_time_t(time_point);
		std::tm* tm = std::gmtime(&time);
		day = tm->tm_mday;
		month = tm->tm_mon + 1;
		year = tm->tm_year + 1900;
	}

	std::string to_string() const {
		return std::to_string(day) + "." + std::to_string(month) + "." + std::to_string(year);
	}

	uint16_t day = 0, month = 0, year = 0;
};

inline bool operator<(const Date& lhs, const Date& rhs) {
	return std::tie(lhs.year, lhs.month, lhs.day) <
		std::tie(rhs.year, rhs.month, rhs.day);
}

inline bool operator>(const Date& lhs, const Date& rhs) {
	return std::tie(lhs.year, lhs.month, lhs.day) >
		std::tie(rhs.year, rhs.month, rhs.day);
}

inline std::string read_file(const std::string& path) {
	std::ifstream file(path);
	std::string data;
	file >> data;
	return data;
}

inline void write_file(const std::string& path, const std::string& data) {
	std::ofstream file(path);
	file << data;
}

std::string pack(const nlohmann::json& j, const std::string& private_key_base64);
nlohmann::json unpack(const std::string& container, const std::string& public_key_base64);

void verify_license_legacy(const std::string& in_path, const std::string& public_key);

}
