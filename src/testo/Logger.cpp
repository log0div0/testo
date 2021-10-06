
#include "Logger.hpp"

thread_local uint32_t trace_level = 0;

Trace::Trace(const std::string fn_name_): fn_name(std::move(fn_name_)) {
	print_message(" begin");
	++trace_level;
}

Trace::~Trace() {
	--trace_level;
	if (std::uncaught_exception()) {
		print_message(" end (stack unwinding");
	} else {
		print_message(" end");
	}
}

void Trace::print_message(const std::string& suffix) {
	std::string message;
	for (size_t i = 0; i < trace_level; ++i) {
		message.push_back('\t');
	}
	message += fn_name;
	message += suffix;
	spdlog::trace(message);
}
