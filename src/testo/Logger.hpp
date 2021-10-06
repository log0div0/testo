
#pragma once

#include <spdlog/spdlog.h>

struct Trace {
	Trace(const std::string fn_name_);
	~Trace();

private:
	void print_message(const std::string& suffix);
	std::string fn_name;
};

#define TRACE() Trace trace(__PRETTY_FUNCTION__)