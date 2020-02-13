
#pragma once

#include <string>

struct Process {
	Process(const std::string& cmd);
	~Process();

	Process(Process&& other);
	Process& operator=(Process&& other);

	int wait();
	bool is_running();
	std::string read();

private:
	FILE* pipe = 0;
	int rc = 0;
};
