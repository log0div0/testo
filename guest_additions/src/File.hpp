
#pragma once

#include <vector>
#include <experimental/filesystem>

void make_directories(const std::experimental::filesystem::path& path);
std::vector<uint8_t> read_file(const std::experimental::filesystem::path& path);
void write_file(const std::experimental::filesystem::path& path, const std::vector<uint8_t>& data);
