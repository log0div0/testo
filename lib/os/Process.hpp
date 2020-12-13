
#pragma once

#include <string>
#include <stdexcept>

#ifdef WIN32
#include <winapi/Process.hpp>
#else
#include <posixapi/Process.hpp>
#endif

namespace os {

struct ProcessError: std::runtime_error {
	using std::runtime_error::runtime_error;
	std::string output;
	int exit_code = 0;
};

#ifdef WIN32
struct Process: winapi::Process {
	using winapi::Process::Process;
	using Base = winapi::Process;
#else
struct Process: posixapi::Process {
	using posixapi::Process::Process;
	using Base = posixapi::Process;
#endif
	static std::pair<std::string, int> exec_no_throw(const std::string& cmd);
	static std::string exec(const std::string& cmd);

	std::string read();
	int wait();

#ifdef WIN32
	bool eof() {
		return _eof;
	}
	bool _eof = false;
#endif
};

}
