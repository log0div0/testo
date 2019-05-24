
#include "Utils.hpp"
#include <sys/types.h>
#include <algorithm>

uint32_t time_to_seconds(const std::string& time) {
	uint32_t seconds = std::stoul(time.substr(0, time.length() - 1));
	if (time[time.length() - 1] == 's') {
		seconds = seconds * 1;
	} else if (time[time.length() - 1] == 'm') {
		seconds = seconds * 60;
	} else if (time[time.length() - 1] == 'h') {
		seconds = seconds * 60 * 60;
	} else {
		throw std::runtime_error("Unknown time specifier"); //should not happen ever
	}

	return seconds;
}

void exec_and_throw_if_failed(const std::string& command) {
	if (std::system(command.c_str())) {
		throw std::runtime_error("Command failed: " + command);
	}
}

#ifdef WIN32

fs::path home_dir() {
	throw std::runtime_error(__FUNCSIG__);
}

#else

#include <pwd.h>
#include <unistd.h>

fs::path home_dir() {
	//struct passwd *pw = getpwuid(getuid());
	//return fs::path(pw->pw_dir);
	return fs::path("/var/lib/libvirt");
}

#endif

fs::path testo_dir() {
	auto res = home_dir();
	res = res / "/testo";
	return res;
}

fs::path flash_drives_img_dir() {
	auto res = home_dir();
	res = res / "/testo/flash_drives/images/";
	return res;
}

fs::path flash_drives_mount_dir() {
	auto res = home_dir();
	res = res / "/testo/flash_drives/mount_point/";
	return res;
}

fs::path scripts_tmp_dir() {
	auto res = home_dir();
	res = res / "/testo/scripts_tmp/";
	return res;
}

std::string file_signature(const fs::path& file) {
	if (!fs::exists(file)) {
		throw std::runtime_error("File " + file.generic_string() + " does not exists");
	}
	auto last_modify_time = std::chrono::system_clock::to_time_t(fs::last_write_time(file));
	return file.filename().generic_string() + std::to_string(last_modify_time);
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

