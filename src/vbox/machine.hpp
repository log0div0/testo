
#pragma once

#include "api.hpp"
#include "string.hpp"

namespace vbox {

struct Machine {
	Machine();
	Machine(IMachine* handle);
	~Machine();

	Machine(const Machine&) = delete;
	Machine& operator=(const Machine&) = delete;

	Machine(Machine&& other);
	Machine& operator=(Machine&& other);

	String name() const;

	IMachine* handle = nullptr;
};

}
