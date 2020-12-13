
#include "Process.hpp"
#include <stdexcept>

namespace posixapi {

Process::Process(const std::string& cmd) {
	pipe = popen(cmd.c_str(), "r");
	if (!pipe) {
		throw std::runtime_error("popen() failed!");
	}
}

Process::~Process() {
	if (pipe) {
		pclose(pipe);
		pipe = 0;
	}
}

bool Process::eof() {
	return feof(pipe);
}

size_t Process::read(uint8_t* data, size_t size) {
	return fread(data, 1, size, pipe);
}

int Process::close() {
	if (pipe == 0) {
		throw std::runtime_error("Process handle already closed");
	}
	int res = pclose(pipe);
	pipe = 0;
	return res;
}

}
