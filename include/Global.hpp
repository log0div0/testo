
#pragma once

#include <API.hpp>
#include <VmController.hpp>
#include <FlashDriveController.hpp>
#include <Node.hpp>
#include <unordered_map>

struct Global {
	Global();
	~Global();

	std::unordered_map<std::string, std::shared_ptr<VmController>> local_vms;
	std::unordered_map<std::string, std::shared_ptr<VmController>> vms;
	std::unordered_map<std::string, std::shared_ptr<FlashDriveController>> fds;
	std::unordered_map<std::string, std::shared_ptr<AST::Snapshot>> snapshots;
	std::unordered_map<std::string, std::shared_ptr<AST::Macro>> macros;

	void setup();
	void cleanup();
	API& api;
};
