
#include "Process.hpp"

namespace os {

std::pair<std::string, int> Process::exec_no_throw(const std::string& cmd) {
	Process process(cmd);
	std::string result;
	while (!process.eof()) {
		result += process.read();
	}
	int rc = process.wait();
	return {result, rc};
}

std::string Process::exec(const std::string& cmd) {
	auto pair = exec_no_throw(cmd);
	if (pair.second) {
		ProcessError error("Command " + cmd + " exits with code " + std::to_string(pair.second) + ". Command output: " + pair.first);
		error.exit_code = pair.second;
		error.output = pair.first;
		throw error;
	}
	return pair.first;
}

std::string Process::read() {
	const int buf_size = 256;
	char buf[buf_size];
	size_t n = Base::read((uint8_t*)buf, buf_size);
#ifdef WIN32
	if (n == 0) {
		_eof = true;
	}
#endif
	return {buf, buf + n};
}

int Process::wait() {
#ifdef WIN32
	Base::wait();
	return get_exit_code();
#else
	return close();
#endif
}

}
