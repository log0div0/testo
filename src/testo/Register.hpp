
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
};
