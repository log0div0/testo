
#pragma once

#include "backends/VmController.hpp"
#include "backends/FlashDriveController.hpp"
#include "Node.hpp"
#include <unordered_map>
#include <list>

struct Register {
	Register() = default;
	~Register();

	std::unordered_map<std::string, std::shared_ptr<VmController>> vmcs;
	std::unordered_map<std::string, std::shared_ptr<FlashDriveController>> fdcs;
	std::unordered_map<std::string, std::shared_ptr<AST::Test>> tests;
	std::unordered_map<std::string, std::shared_ptr<AST::Macro>> macros;

	std::set<std::shared_ptr<Controller>> get_all_controllers(std::shared_ptr<AST::Test> test) const;
	std::set<std::shared_ptr<VmController>> get_all_vmcs(std::shared_ptr<AST::Test> test) const;
	std::list<std::shared_ptr<AST::Test>> get_test_path(std::shared_ptr<AST::Test> test) const;
};