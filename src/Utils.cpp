
#include <Utils.hpp>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

void backtrace(std::ostream& stream, const std::exception& error, size_t n) {
	stream << n << ". " << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << std::endl;
		backtrace(stream, error, n + 1);
	} catch(...) {
		stream << std::endl;
		stream << n << ". " << "[Unknown exception type]";
	}
}

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

fs::path home_dir() {
	struct passwd *pw = getpwuid(getuid());
	return fs::path(pw->pw_dir);
}

fs::path flash_drives_img_dir() {
	auto res = home_dir();
	res += "/testo/vbox/flash_drives/images/";
	return res;
}

fs::path flash_drives_mount_dir() {
	auto res = home_dir();
	res += "/testo/vbox/flash_drives/mount_point/";
	return res;
}

fs::path scripts_tmp_dir() {
	auto res = home_dir();
	res += "/testo/vbox/scripts_tmp/";
	return res;
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
