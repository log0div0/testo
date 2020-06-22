
#pragma once

#include "backends/VmController.hpp"
#include "backends/FlashDriveController.hpp"
#include "backends/NetworkController.hpp"
#include "backends/Environment.hpp"
#include "StackEntry.hpp"
#include "Node.hpp"
#include <unordered_map>
#include <list>

struct FlashDriveControllerRequest {
	FlashDriveControllerRequest() = delete;
	FlashDriveControllerRequest(const nlohmann::json& config): config(config) {}

	std::shared_ptr<FlashDriveController> get_fdc();

	nlohmann::json config;
	std::shared_ptr<FlashDriveController> fdc = nullptr;
};

struct Register;

struct VmControllerRequest {
	VmControllerRequest() = delete;
	VmControllerRequest(const nlohmann::json& config, std::shared_ptr<Register> reg): config(config), reg(reg) {}

	std::shared_ptr<VmController> get_vmc();

	nlohmann::json config;
	std::shared_ptr<Register> reg = nullptr;
	std::shared_ptr<VmController> vmc = nullptr;
};

struct Register {
	Register() = default;
	~Register() {
		for (auto fdc_request: fdc_requests) {
			if (fdc_request.second.fdc->fd->is_mounted()) {
				fdc_request.second.fdc->fd->umount();
			}
		}
	}

	std::unordered_map<std::string, VmControllerRequest> vmc_requests;
	std::unordered_map<std::string, FlashDriveControllerRequest> fdc_requests;
	std::unordered_map<std::string, std::shared_ptr<NetworkController>> netcs;
	std::unordered_map<std::string, std::shared_ptr<AST::Test>> tests;
	std::unordered_map<std::string, std::shared_ptr<AST::Controller>> controllers;
	std::unordered_map<std::string, std::shared_ptr<AST::MacroAction>> macros_action;
	std::unordered_map<std::string, std::shared_ptr<AST::Param>> param_nodes;
	std::unordered_map<std::string, std::string> params;

	std::vector<StackEntry> local_vars;
};
