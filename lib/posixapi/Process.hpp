
#pragma once

#include <string>

namespace posixapi {

struct Process {
	Process(const std::string& cmd);
	~Process();

	Process(Process&& other);
	Process& operator=(Process&& other);

	bool eof();
	size_t read(uint8_t* data, size_t size);
	int close();

private:
	FILE* pipe = 0;
};

}
