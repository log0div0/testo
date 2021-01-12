
#pragma once

#include <iostream>
#include <string>
#include <list>
#include <nlohmann/json.hpp>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

void fs_copy(const fs::path& from, const fs::path& to);

bool check_if_time_interval(const std::string& time);
uint32_t time_to_milliseconds(const std::string& time);

extern uint64_t content_cksum_maxsize;
std::string file_signature(const fs::path& file);
std::string directory_signature(const fs::path& dir);

bool is_number(const std::string& s);

bool is_mac_correct(const std::string& mac);
std::string normalized_mac(const std::string& mac);

void replace_all(std::string& str, const std::string& from, const std::string& to);

nlohmann::json read_metadata_file(const fs::path& file);
void write_metadata_file(const fs::path& file, const nlohmann::json& metadata);
std::string get_metadata(const fs::path& file, const std::string& key);

std::pair<int, int> parse_usb_addr(const std::string& addr);

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
