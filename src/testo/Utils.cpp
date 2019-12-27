
#include "Utils.hpp"
#include <sys/types.h>
#include <fstream>

uint32_t time_to_milliseconds(const std::string& time) {
	uint32_t milliseconds;
	if (time[time.length() - 2] == 'm') {
		milliseconds = std::stoul(time.substr(0, time.length() - 2));
	} else if (time[time.length() - 1] == 's') {
		milliseconds = std::stoul(time.substr(0, time.length() - 1));
		milliseconds = milliseconds * 1000;
	} else if (time[time.length() - 1] == 'm') {
		milliseconds = std::stoul(time.substr(0, time.length() - 1));
		milliseconds = milliseconds * 1000 * 60;
	} else if (time[time.length() - 1] == 'h') {
		milliseconds = std::stoul(time.substr(0, time.length() - 1));
		milliseconds = milliseconds * 1000 * 60 * 60;
	} else {
		throw std::runtime_error("Unknown time specifier"); //should not happen ever
	}

	return milliseconds;
}

void exec_and_throw_if_failed(const std::string& command) {
	if (std::system(command.c_str())) {
		throw std::runtime_error("Command failed: " + command);
	}
}

std::string file_signature(const fs::path& file) {
	if (!fs::exists(file)) {
		return file.filename().generic_string() + "not exists";
	}

	if(fs::file_size(file) > 1048576) { //1Mb
		auto last_modify_time = std::chrono::system_clock::to_time_t(fs::last_write_time(file));
		return file.filename().generic_string() + std::to_string(last_modify_time);
	} else {
		std::ifstream f(file.generic_string());
		if (!f) {
			return file.filename().generic_string() + "Can't open";
		}
		std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		std::hash<std::string> h;
		return file.filename().generic_string() + std::to_string(h(str));
	}

}

std::string directory_signature(const fs::path& dir) {
	std::string result("");
	for (auto& file: fs::directory_iterator(dir)) {
		if (fs::is_regular_file(file)) {
			result += file_signature(file);
		} else if (fs::is_directory(file)) {
			result += directory_signature(file);
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(file).generic_string());
		}
	}

	auto last_modify_time = std::chrono::system_clock::to_time_t(fs::last_write_time(dir));
	result += std::to_string(last_modify_time);
	return result;
}

//NOTE: this check is very, very rough
bool is_mac_correct(const std::string& mac) {
	int k = 0, s = 0;

	for (size_t i = 0; i < mac.length(); i++) {
		if (isxdigit(mac[i])) {
			k++;
		} else if (mac[i] == ':') {
			if (k == 0 || k / 2 - 1 != s) {
				break;
			}
			++s;
		} else {
			s = -1;
		}
	}

	return (k == 12 && (s == 5 || s == 0));
}

std::string normalized_mac(const std::string& mac) {
	std::string result;
	for (size_t i = 0; i < mac.length(); i++) {
		if (mac[i] == ':') {
			continue;
		}
		result += mac[i];
	}

	return result;
}

bool is_number(const std::string& s) {
	return !s.empty() && std::find_if(s.begin(),
		s.end(), [](char c) { return !isdigit(c); }) == s.end();
}

void replace_all(std::string& str, const std::string& from, const std::string& to) {
	if(from.empty())
		return;
	size_t start_pos = 0;
	while((start_pos = str.find(from, start_pos)) != std::string::npos) {
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
	}
}

nlohmann::json read_metadata_file(const fs::path& file) {
	std::ifstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't read metadata file " + file.generic_string());
	}

	nlohmann::json result = nlohmann::json::parse(metadata_file_stream);
	metadata_file_stream.close();
	return result;
}


void write_metadata_file(const fs::path& file, const nlohmann::json& metadata) {
	std::ofstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't write metadata file " + file.generic_string());
	}

	metadata_file_stream << metadata;
	metadata_file_stream.close();
}

std::string get_metadata(const fs::path& file, const std::string& key) {
	auto metadata = read_metadata_file(file);
	if (!metadata.count(key)) {
		throw std::runtime_error("Requested key is not present in metadata");
	}
	return metadata.at(key).get<std::string>();
}
