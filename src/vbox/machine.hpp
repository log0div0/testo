
#pragma once

#include "api.hpp"
#include <string>

namespace vbox {

struct Machine {
	Machine();
	Machine(IMachine* handle);
	~Machine();

	Machine(const Machine&) = delete;
	Machine& operator=(const Machine&) = delete;

	Machine(Machine&& other);
	Machine& operator=(Machine&& other);

	std::string name() const;

	using Iface = IMachine;
	IMachine* handle = nullptr;
};

}
