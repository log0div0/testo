
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

	std::chrono::system_clock::time_point to_time_point() const {
		std::tm timeinfo = std::tm();
		timeinfo.tm_year = year;
		timeinfo.tm_mon = month;
		timeinfo.tm_mday = day;
		std::time_t tt = std::mktime(&timeinfo);
		return std::chrono::system_clock::from_time_t(tt);
	}

	uint16_t day = 0, month = 0, year = 0;
};

void sign_license(const std::string& license_path, const std::string& private_key);
std::string verify_license(const std::string& license_path, const std::string& public_key);
