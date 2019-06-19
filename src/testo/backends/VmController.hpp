
#pragma once

#include "VM.hpp"

struct VmController {
	VmController() = delete;
	VmController(std::shared_ptr<VM> vm): vm(vm) {}


	std::shared_ptr<VM> vm;
};
