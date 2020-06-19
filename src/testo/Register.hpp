
#pragma once

#include "backends/VmController.hpp"
#include "backends/FlashDriveController.hpp"
#include "backends/NetworkController.hpp"
#include "StackEntry.hpp"
#include "Node.hpp"
#include <unordered_map>
#include <list>

struct Register {
	Register() = default;
	~Register();

	std::unordered_map<std::string, std::shared_ptr<VmController>> vmcs;
	std::unordered_map<std::string, std::shared_ptr<FlashDriveController>> fdcs;
	std::unordered_map<std::string, std::shared_ptr<NetworkController>> netcs;
	std::unordered_map<std::string, std::shared_ptr<AST::Test>> tests;
	std::unordered_map<std::string, std::shared_ptr<AST::Controller>> controllers;
	std::unordered_map<std::string, std::shared_ptr<AST::MacroAction>> macros_action;
	std::unordered_map<std::string, std::shared_ptr<AST::Param>> param_nodes;
	std::unordered_map<std::string, std::string> params;

	std::vector<StackEntry> local_vars;

	std::set<std::shared_ptr<Controller>> get_all_controllers(std::shared_ptr<AST::Test> test) const;
	std::set<std::shared_ptr<VmController>> get_all_vmcs(std::shared_ptr<AST::Test> test) const;
	std::set<std::shared_ptr<NetworkController>> get_all_netcs(std::shared_ptr<AST::Test> test) const;
	std::set<std::shared_ptr<FlashDriveController>> get_all_fdcs(std::shared_ptr<AST::Test> test) const;
	std::list<std::shared_ptr<AST::Test>> get_test_path(std::shared_ptr<AST::Test> test) const;

private:
	std::set<std::shared_ptr<FlashDriveController>> extract_fdcs_from_action(std::shared_ptr<AST::IAction> action) const;
};
