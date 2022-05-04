
#pragma once

#include <iostream>
#include <string>
#include <list>
#include <nlohmann/json.hpp>

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

void fs_copy(const fs::path& from, const fs::path& to);

extern uint64_t content_cksum_maxsize;
std::string file_signature(const fs::path& file);
std::string pretty_files_signature(const fs::path& item, size_t depth = 0);

bool is_number(const std::string& s);

bool is_mac_correct(const std::string& mac);
std::string normalized_mac(const std::string& mac);

void replace_all(std::string& str, const std::string& from, const std::string& to);

std::pair<int, int> parse_usb_addr(const std::string& addr);

template <typename T>
void concat_unique(std::list<T>& left, const std::list<T>& right) {

	for (auto it_right: right) {
		bool already_included = false;
		for (auto it_left: left) {
			if (it_left == it_right) {
				already_included = true;
				break;
			}
		}
		if (!already_included) {
			left.push_back(it_right);
		}
	}
}

std::string generate_uuid_v4();
