
#pragma once

#include "api.hpp"

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

	IProgress* handle = nullptr;
};

}
