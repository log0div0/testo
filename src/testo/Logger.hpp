
#pragma once

#include <spdlog/spdlog.h>

struct Trace {
	Trace(const std::string fn_name_): fn_name(std::move(fn_name_)) {
		spdlog::trace(fn_name + " begin");
	}
	~Trace() {
		if (std::uncaught_exception()) {
			spdlog::trace(fn_name + " end (stack unwinding");
		} else {
			spdlog::trace(fn_name + " end");
		}
	}

private:
	std::string fn_name;
};

#define TRACE() Trace trace(__PRETTY_FUNCTION__)