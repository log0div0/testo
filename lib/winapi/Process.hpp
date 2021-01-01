
#pragma once

#include "Pipe.hpp"
#include <string>
#include <map>

namespace winapi {

struct Process {
	Process(const std::string& cmd, const std::map<std::string, std::string>* env_vars = nullptr);
	~Process();

	Process(Process&& other);
	Process& operator=(Process&& other);

	size_t read(uint8_t* data, size_t size);
	void wait();
	int get_exit_code();

private:
	Pipe in_write;
	Pipe out_read;
	HANDLE process = NULL;
};

}