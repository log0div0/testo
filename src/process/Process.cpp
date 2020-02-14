
#include "Process.hpp"

#include <stdexcept>
#include <array>

std::string Process::exec(const std::string& cmd) {
	Process process(cmd);
	std::string result;
	while (process.is_running()) {
		result += process.read();
	}
	return result;
}

Process::Process(const std::string& cmd) {
	pipe = popen(cmd.c_str(), "r");
	if (!pipe) {
		throw std::runtime_error("popen() failed!");
	}
}

Process::~Process() {
	wait();
}

int Process::wait() {
	if (pipe) {
		rc = pclose(pipe);
		pipe = 0;
	}
	return rc;
}

bool Process::is_running() {
	return !feof(pipe);
}

std::string Process::read() {
	std::array<char, 256> buffer;
	if (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
		return buffer.data();
	} else {
		return {};
	}
}
