
#pragma once

#include <iostream>
#include <string>
#include <experimental/filesystem>
#include "pugixml/pugixml.hpp"

namespace fs = std::experimental::filesystem;

static void backtrace(std::ostream& stream, const std::exception& error, size_t n) {
	stream << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << ":\n";
		for (size_t i = 0; i < n; i++) {
			stream << "\t";
		}
		stream << "-";
		backtrace(stream, error, n + 1);
	} catch(...) {
		stream << std::endl;
		stream << "[Unknown exception type]";
	}
}

uint32_t time_to_seconds(const std::string& time);
void exec_and_throw_if_failed(const std::string& command);
fs::path home_dir();
fs::path testo_dir();
fs::path flash_drives_img_dir();
fs::path flash_drives_mount_dir();
fs::path scripts_tmp_dir();

std::string file_signature(const fs::path& file);
std::string directory_signature(const fs::path& dir);

bool is_number(const std::string& s);

bool is_mac_correct(const std::string& mac);
std::string normalized_mac(const std::string& mac);

void replace_all(std::string& str, const std::string& from, const std::string& to);

struct xml_string_writer: pugi::xml_writer
{
	std::string result;

	virtual void write(const void* data, size_t size)
	{
		result.append(static_cast<const char*>(data), size);
	}
};


inline std::string node_to_string(pugi::xml_node node)
{
	xml_string_writer writer;
	node.print(writer);

	return writer.result;
}

inline std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error, 1);
	return stream;
}
