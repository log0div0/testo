
#include <coro/CheckPoint.h>
#include "Utils.hpp"
#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <iomanip>
#include <os/File.hpp>

void fs_copy_file(const fs::path& from, const fs::path& to) {
	os::File source = os::File::open_for_read(from);
	os::File dest = os::File::open_for_write(to);

	uint8_t buf[8192];
	size_t size;

	while ((size = source.read(buf, sizeof(buf))) > 0) {
		dest.write(buf, size);
		coro::CheckPoint();
	}
}

void fs_copy(const fs::path& from, const fs::path& to) {

	if (!fs::exists(from)) {
		throw std::runtime_error("Fs_copy error: \"from\" path " + from.generic_string() + " does not exist");
	}

	if (fs::exists(to) && fs::equivalent(from, to)) {
		throw std::runtime_error("Fs_copy error: \"from\" path " + from.generic_string() + " and \"to\" path " + to.generic_string() + " are equalent");
	}

	if (!fs::is_regular_file(to) && !fs::is_directory(to) && fs::exists(to)) {
		throw std::runtime_error("Fs_copy: Unsupported type of file: " + to.generic_string());
	}

	if (fs::is_directory(from) && fs::is_regular_file(to)) {
		throw std::runtime_error("Fs_copy: can't copy a directory " + from.generic_string() + " to a regular file " + to.generic_string());
	}

	//if from is a regular file
	if (fs::is_regular_file(from)) {
		if (fs::is_directory(to)) {
			fs_copy_file(from, to / from.filename());
		} else {
			if (!fs::exists(to.parent_path()) && !fs::create_directories(to.parent_path())) {
				throw std::runtime_error("Fs_copy error: can't create directory " + to.parent_path().generic_string());
			}
			fs_copy_file(from, to);
		}
	} else if (fs::is_directory(from)) {
		if (!fs::exists(to.parent_path()) && !fs::create_directories(to)) {
			throw std::runtime_error("Fs_copy error: can't create directory " + to.generic_string());
		}
		for (auto& directory_entry: fs::directory_iterator(from)) {
			fs_copy(directory_entry.path(), to / directory_entry.path().filename());
		}

	} else {
		throw std::runtime_error("Fs_copy: Unsupported type of file: " + from.generic_string());
	}
}

bool check_if_time_interval(const std::string& time) {
	std::string number;

	size_t i = 0;

	for (; i < time.length(); ++i) {
		if (isdigit(time[i])) {
			number += time[i];
		} else {
			break;
		}
	}

	if (!number.length()) {
		return false;
	}

	if (time[i] == 's' || time[i] == 'h') {
		return (i == time.length() - 1);
	}

	if (time[i] == 'm') {
		if (i == time.length() - 1) {
			return true;
		}

		if (time.length() > i + 2) {
			return false;
		}
		return time[i + 1] == 's';
	}

	return false;

}

std::chrono::milliseconds time_to_milliseconds(const std::string& time) {
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

	return std::chrono::milliseconds(milliseconds);
}

uint64_t content_cksum_maxsize = 1;

template <typename T>
std::string to_hex(T i) {
	std::stringstream stream;
	stream << "0x"
		<< std::setfill ('0') << std::setw(sizeof(T)*2)
		<< std::hex << i;
	return stream.str();
}

std::string file_signature(const fs::path& file) {
	if (!fs::exists(file)) {
		return "NOT_EXISTS";
	}

	if(fs::file_size(file) > (content_cksum_maxsize * 1024 * 1024)) {
		auto time = fs::last_write_time(file);
		auto last_modify_time = decltype(time)::clock::to_time_t(time);
		char buf[32] = {};
		std::strftime(buf, 32, "%Y.%m.%d %H:%m:%S", std::localtime(&last_modify_time));
		return buf;
	} else {
		std::ifstream f(file.generic_string());
		if (!f) {
			return "FAILED_TO_OPEN";
		}
		std::string str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		std::hash<std::string> h;
		return to_hex(h(str));
	}

}

std::string pretty_files_signature(const fs::path& path, size_t depth) {
	std::string result;
	for (size_t i = 0; i < depth; ++i) {
		result.push_back('\t');
	}
	if (fs::is_regular_file(path)) {
		result += file_signature(path) + " " + path.filename().generic_string();
	} else if (fs::is_directory(path)) {
		result += path.filename().generic_string() + " {\n";
		std::vector<fs::path> paths;
		for (auto& file: fs::directory_iterator(path)) {
			paths.push_back(file);
		}
		std::sort(paths.begin(), paths.end());
		for (auto& file: paths) {
			result += pretty_files_signature(file, depth + 1) + "\n";
		}
		for (size_t i = 0; i < depth; ++i) {
			result.push_back('\t');
		}
		result += "}";
	} else {
		throw std::runtime_error("Unknown type of file: " + fs::path(path).generic_string());
	}
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
	if (s.empty()) {
		return false;
	}

	auto begin = s.begin();
	if (s[0] == '-') {
		begin++;
	}

	return std::find_if(begin, s.end(), [](char c) { return !isdigit(c); }) == s.end();
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

std::pair<int, int> parse_usb_addr(const std::string& addr) {
	std::stringstream ss(addr);
	std::string segment;
	std::vector<std::string> seglist;

	while(std::getline(ss, segment, '-'))
	{
		seglist.push_back(segment);
	}

	if (seglist.size() != 2) {
		throw std::runtime_error("Parsing usb addr error");
	}

	if (!is_number(seglist[0])) {
		throw std::runtime_error("Parsing usb addr error");
	}

	if (!is_number(seglist[1])) {
		throw std::runtime_error("Parsing usb addr error");
	}

	int bus_id = std::stoi(seglist[0]);
	int dev_id = std::stoi(seglist[1]);

	if (bus_id < 0) {
		throw std::runtime_error("Parsing usb addr error");
	}

	if (dev_id < 0) {
		throw std::runtime_error("Parsing usb addr error");
	}

	return {bus_id, dev_id};
}

std::string generate_uuid_v4() {
	std::random_device              rd;
	std::mt19937                    gen(rd());
	std::uniform_int_distribution<> dis(0, 15);
	std::uniform_int_distribution<> dis2(8, 11);

	std::stringstream ss;
	int i;
	ss << std::hex;
	for (i = 0; i < 8; i++) {
		ss << dis(gen);
	}
	ss << "-";
	for (i = 0; i < 4; i++) {
		ss << dis(gen);
	}
	ss << "-4";
	for (i = 0; i < 3; i++) {
		ss << dis(gen);
	}
	ss << "-";
	ss << dis2(gen);
	for (i = 0; i < 3; i++) {
		ss << dis(gen);
	}
	ss << "-";
	for (i = 0; i < 12; i++) {
		ss << dis(gen);
	};
	return ss.str();
}
