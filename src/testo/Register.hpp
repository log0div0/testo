
#pragma once

#include "backends/VmController.hpp"
#include "backends/FlashDriveController.hpp"
#include "Node.hpp"
#include <unordered_map>

struct Register {
	Register() = default;
	~Register();

	std::unordered_map<std::string, std::shared_ptr<VmController>> vms;
	std::unordered_map<std::string, std::shared_ptr<FlashDriveController>> fds;
	std::unordered_map<std::string, std::shared_ptr<AST::Test>> tests;
	std::unordered_map<std::string, std::shared_ptr<AST::Macro>> macros;

	std::set<std::shared_ptr<VmController>> get_all_vms(std::shared_ptr<AST::Test> test) const;
	std::vector<std::shared_ptr<AST::Test>> get_test_path(std::shared_ptr<AST::Test> test) const;
};