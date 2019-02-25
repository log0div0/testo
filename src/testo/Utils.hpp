
#pragma once

#include <iostream>
#include <string>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

void backtrace(std::ostream& stream, const std::exception& error, size_t n);
uint32_t time_to_seconds(const std::string& time);
void exec_and_throw_if_failed(const std::string& command);
fs::path home_dir();
fs::path testo_dir();
fs::path flash_drives_img_dir();
fs::path flash_drives_mount_dir();
fs::path scripts_tmp_dir();


bool is_number(const std::string& s);

bool is_mac_correct(const std::string& mac);
std::string normalized_mac(const std::string& mac);

void remove_newlines(std::string& str);

inline std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error, 1);
	return stream;
}
