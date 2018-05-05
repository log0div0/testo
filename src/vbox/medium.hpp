
#pragma once

#include "api.hpp"

namespace vbox {

struct Medium {
	Medium(IMedium* handle);
	~Medium();

	Medium(const Medium&) = delete;
	Medium& operator=(const Medium&) = delete;
	Medium(Medium&& other);
	Medium& operator=(Medium&& other);

	std::string name() const;

	IMedium* handle = nullptr;
};

}
