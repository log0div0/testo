
#include "Process.hpp"
#ifdef WIN32
#include "winapi.hpp"
#endif

#include <iostream>
#include <array>

std::pair<std::string, int> Process::exec_no_throw(const std::string& cmd) {
	// std::cout << "PROCESS: " << cmd << std::endl;
	Process process(cmd);
	std::string result;
	while (!process.eof()) {
		result += process.read();
	}
	int rc = process.wait();
	// std::cout << "ERROR CODE: " << rc << std::endl;
	return {result, rc};
}

std::string Process::exec(const std::string& cmd) {
	auto pair = exec_no_throw(cmd);
	if (pair.second) {
		ProcessError error("Command " + cmd + " exits with code " + std::to_string(pair.second));
		error.exit_code = pair.second;
		error.output = pair.first;
		throw error;
	}
	return pair.first;
}

bool Process::is_succeeded(const std::string& cmd) {
	return exec_no_throw(cmd).second == 0;
}

bool Process::is_failed(const std::string& cmd) {
	return exec_no_throw(cmd).second != 0;
}

#if defined(__linux__) || defined(__APPLE__)
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

bool Process::eof() {
	return feof(pipe);
}

std::string Process::read() {
	std::array<char, 256> buffer;
	if (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
		return buffer.data();
	} else {
		return {};
	}
}
#endif

#ifdef WIN32

std::pair<Pipe, Pipe> Pipe::create() {
	SECURITY_ATTRIBUTES saAttr = {};
	saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
	saAttr.bInheritHandle = TRUE;
	saAttr.lpSecurityDescriptor = NULL;

	HANDLE read = NULL;
	HANDLE write = NULL;

	if (!CreatePipe(&read, &write, &saAttr, 0)) {
		throw std::runtime_error("CreatePipe failed");
	}

	return {read, write};
}

Pipe::Pipe(HANDLE handle_): handle(handle_) {

}

Pipe::~Pipe() {
	if (handle) {
		CloseHandle(handle);
		handle = NULL;
	}
}


Pipe::Pipe(Pipe&& other): handle(other.handle) {
	other.handle = NULL;
}

Pipe& Pipe::operator=(Pipe&& other) {
	std::swap(handle, other.handle);
	return *this;
}


void Pipe::setHandleInformation() {
	if (!SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0)) {
		throw std::runtime_error("SetHandleInformation failed");
	}
}

Process::Process(const std::string& cmd) {
	Pipe in_read;
	Pipe out_write;
	std::tie(in_read, in_write) = Pipe::create();
	std::tie(out_read, out_write) = Pipe::create();
	in_write.setHandleInformation();
	out_read.setHandleInformation();

	STARTUPINFO siStartInfo = {};
	siStartInfo.cb = sizeof(STARTUPINFO);
	siStartInfo.hStdError = out_write.handle;
	siStartInfo.hStdOutput = out_write.handle;
	siStartInfo.hStdInput = in_read.handle;
	siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

	PROCESS_INFORMATION info = {};

	std::wstring wcmd = winapi::utf8_to_utf16(cmd);
	bool success = CreateProcess(
		NULL,
		&wcmd[0],
		NULL,
		NULL,
		TRUE,
		CREATE_NO_WINDOW,
		NULL,
		NULL,
		&siStartInfo,
		&info);

	if (!success) {
		DWORD dwErrVal = GetLastError();
		std::error_code ec(dwErrVal, std::system_category());
		throw std::system_error(ec, "CreateProcess failed");
	}

	process = info.hProcess;
	CloseHandle(info.hThread);
}

Process::~Process() {
	if (process) {
		wait();
		CloseHandle(process);
		process = NULL;
	}
}

int Process::wait() {
	if (WaitForSingleObject(process, INFINITE) == WAIT_FAILED) {
		throw std::runtime_error("WaitForSingleObject failed");
	}
	DWORD exit_code;
	if (!GetExitCodeProcess(process, &exit_code)) {
		throw std::runtime_error("GetExitCodeProcess failed");
	}
	return exit_code;
}

bool Process::eof() {
	return _eof;
}

std::string Process::read() {
	std::array<char, 256> buffer;
	DWORD bytes_read = 0;
	auto result = ReadFile(out_read.handle, buffer.data(), buffer.size() - 1, &bytes_read, NULL);
	if (!result || (bytes_read == 0)) {
		_eof = true;
	}
	buffer[bytes_read] = 0;
	return buffer.data();
}

#endif
