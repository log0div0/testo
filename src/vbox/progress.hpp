
#pragma once

#include "virtual_box_error_info.hpp"

namespace vbox {

struct Progress {
	Progress(IProgress* handle);
	~Progress();

	Progress(const Progress&) = delete;
	Progress& operator=(const Progress&) = delete;
	Progress(Progress&& other);
	Progress& operator=(Progress&& other);

	std::string description() const;
	void wait_for_completion(long timeout = -1) const;
	void wait_and_throw_if_failed() const;
	LONG result_code() const;
	VirtualBoxErrorInfo error_info() const;

	IProgress* handle = nullptr;
};

}
