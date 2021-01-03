
#include "Process.hpp"
#include "Functions.hpp"
#include <stdexcept>
#include <vector>
#include <system_error>

namespace winapi {

Process::Process(const std::string& cmd, const std::map<std::string, std::string>* env_vars) {
	Pipe in_read;
	Pipe out_write;
	std::tie(in_read, in_write) = Pipe::create();
	std::tie(out_read, out_write) = Pipe::create();
	in_write.setHandleInformation(HANDLE_FLAG_INHERIT, 0);
	out_read.setHandleInformation(HANDLE_FLAG_INHERIT, 0);

	STARTUPINFO siStartInfo = {};
	siStartInfo.cb = sizeof(STARTUPINFO);
	siStartInfo.hStdError = out_write.handle;
	siStartInfo.hStdOutput = out_write.handle;
	siStartInfo.hStdInput = in_read.handle;
	siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

	PROCESS_INFORMATION info = {};

	std::vector<wchar_t> environment;
	if (env_vars) {
		for (auto& kv: *env_vars) {
			std::wstring key = winapi::utf8_to_utf16(kv.first);
			std::wstring value = winapi::utf8_to_utf16(kv.second);
			environment.insert(environment.end(), key.begin(), key.end());
			environment.push_back(L'=');
			environment.insert(environment.end(), value.begin(), value.end());
			environment.push_back(L'\0');
		}
		environment.push_back(L'\0');
	}

	std::wstring wcmd = winapi::utf8_to_utf16(cmd);
	bool success = CreateProcess(
		NULL,
		&wcmd[0],
		NULL,
		NULL,
		TRUE,
		CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
		env_vars ? environment.data() : NULL,
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

void Process::wait() {
	if (WaitForSingleObject(process, INFINITE) == WAIT_FAILED) {
		throw std::runtime_error("WaitForSingleObject failed");
	}
}

int Process::get_exit_code() {
	DWORD exit_code;
	if (!GetExitCodeProcess(process, &exit_code)) {
		throw std::runtime_error("GetExitCodeProcess failed");
	}
	return exit_code;
}

size_t Process::read(uint8_t* data, size_t size) {
	DWORD bytes_read = 0;
	bool success = ReadFile(out_read.handle, data, size, &bytes_read, NULL);
	if (!success) {
		auto error_code = GetLastError();
		if (error_code != ERROR_BROKEN_PIPE) {
			throw std::runtime_error("ReadFile (Process::read) failed:" + std::to_string(error_code));
		}
	}
	return bytes_read;
}

}
