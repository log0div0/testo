
#pragma once

#include <string>
#include <regex>
#include <chrono>

struct Date {
	static Date from_string(const std::string& str) {
		std::regex regex(R"((\d+).(\d+).(\d+))");
		std::cmatch match;
		if (!std::regex_match(str.data(), match, regex)) {
			throw std::runtime_error("Invalid date format");
		}
		Date date;
		date.day = stoi(match[1]);
		date.month = stoi(match[2]);
		date.year = stoi(match[3]);

		if (date.month > 12) {
			throw std::runtime_error("Invalid month number");
		}
		if (date.day > 31) {
			throw std::runtime_error("Invalid day number");
		}

		return date;
	}

	static Date now() {
		std::time_t time = std::time(nullptr);
		std::tm* tm = std::localtime(&time);
		Date date;
		date.day = tm->tm_mday;
		date.month = tm->tm_mon + 1;
		date.year = tm->tm_year + 1900;
		return date;
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

void sign_license(const std::string& license_path, const std::string& private_key);
std::string verify_license(const std::string& license_path, const std::string& public_key);
