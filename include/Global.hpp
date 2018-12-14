
#pragma once

#include <API.hpp>
#include <QemuVmController.hpp>
#include <VboxVmController.hpp>
#include <FlashDriveController.hpp>
#include <Node.hpp>
#include <unordered_map>

struct Global {

#ifdef QEMU
	using VmController = QemuVmController;
#else
	using VmController = VboxVmController;
#endif

	Global();
	~Global();

	std::unordered_map<std::string, std::shared_ptr<VmController>> local_vms;
	std::unordered_map<std::string, std::shared_ptr<VmController>> vms;
	std::unordered_map<std::string, std::shared_ptr<FlashDriveController>> fds;
	std::unordered_map<std::string, std::shared_ptr<AST::Snapshot>> snapshots;

	void setup();
	void cleanup();
	API& api;
};
