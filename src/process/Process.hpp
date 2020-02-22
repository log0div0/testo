
#pragma once

#include <string>
#include <stdexcept>

#ifdef WIN32

#include <Windows.h>

struct Pipe {
	static std::pair<Pipe, Pipe> create();

	Pipe() = default;
	Pipe(HANDLE handle_);
	~Pipe();

	Pipe(Pipe&& other);
	Pipe& operator=(Pipe&& other);

	void setHandleInformation();

	HANDLE handle = NULL;
};
#endif

struct ProcessError: std::runtime_error {
	using std::runtime_error::runtime_error;
	std::string output;
	int exit_code = 0;
};

struct Process {
	static std::string exec(const std::string& cmd);

	Process(const std::string& cmd);
	~Process();

	Process(Process&& other);
	Process& operator=(Process&& other);

	int wait();
	bool eof();
	std::string read();

private:

#if defined(__linux__) || defined(__APPLE__)
	int rc = 0;
	FILE* pipe = 0;
#endif

#ifdef WIN32
	Pipe in_write;
	Pipe out_read;
	HANDLE process = NULL;
	bool _eof = false;
#endif
};
