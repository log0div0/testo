
#pragma once

#include "VmController.hpp"
#include "FlashDriveController.hpp"
#include "Node.hpp"
#include <unordered_map>

struct Register {
	Register() = default;
	~Register();

	std::unordered_map<std::string, std::shared_ptr<VmController>> vms;
	std::unordered_map<std::string, std::shared_ptr<FlashDriveController>> fds;
	std::unordered_map<std::string, std::shared_ptr<AST::Test>> tests;
	std::unordered_map<std::string, std::shared_ptr<AST::Macro>> macros;
};