
#pragma once

#include <iostream>
#include <string>
#include <list>
#include <experimental/filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::experimental::filesystem;

static void backtrace(std::ostream& stream, const std::exception& error) {
	stream << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << ":\n\t-";
		backtrace(stream, error);
	} catch(...) {
		stream << std::endl;
		stream << "[Unknown exception type]";
	}
}

uint32_t time_to_milliseconds(const std::string& time);
void exec_and_throw_if_failed(const std::string& command);

std::string file_signature(const fs::path& file);
std::string directory_signature(const fs::path& dir);
std::string timestamp_signature(const std::string& timestamp);

bool is_number(const std::string& s);

bool is_mac_correct(const std::string& mac);
std::string normalized_mac(const std::string& mac);

void replace_all(std::string& str, const std::string& from, const std::string& to);

nlohmann::json read_metadata_file(const fs::path& file);
void write_metadata_file(const fs::path& file, const nlohmann::json& metadata);
std::string get_metadata(const fs::path& file, const std::string& key);

template <typename T>
void concat_unique(std::list<T>& left, const std::list<T>& right) {

	for (auto it_right: right) {
		bool already_included = false;
		for (auto it_left: left) {
			if (it_left == it_right) {
				already_included = true;
			}
		}
		if (!already_included) {
			left.push_back(it_right);
		}
	}
}

inline std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error);
	return stream;
}
