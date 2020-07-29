
#include "ModeClean.hpp"
#include "Utils.hpp"
#include "backends/Environment.hpp"
#include "IR/Network.hpp"
#include "IR/FlashDrive.hpp"
#include "IR/Machine.hpp"

int clean_mode(const CleanModeArgs& args) {
	//cleanup networks
	for (auto& network_folder: fs::directory_iterator(env->network_metadata_dir())) {
		for (auto& file: fs::directory_iterator(network_folder)) {
			try {
				if (fs::path(file).filename() == fs::path(network_folder).filename()) {
					IR::Network network;
					network.config = nlohmann::json::parse(get_metadata(file, "network_config"));
					if (network.nw()->prefix() == args.prefix) {
						network.undefine();
						std::cout << "Deleted network " << network.nw()->id() << std::endl;
						break;
					}
				}
			} catch (const std::exception& error) {
				std::cerr << "Couldn't remove network " << fs::path(file).filename() << std::endl;
				std::cerr << error << std::endl;
			}

		}
	}

	//cleanup flash drives
	for (auto& flash_drive_folder: fs::directory_iterator(env->flash_drives_metadata_dir())) {
		for (auto& file: fs::directory_iterator(flash_drive_folder)) {
			try {
				if (fs::path(file).filename() == fs::path(flash_drive_folder).filename()) {
					IR::FlashDrive flash_drive;
					flash_drive.config = nlohmann::json::parse(get_metadata(file, "fd_config"));
					if (flash_drive.fd()->prefix() == args.prefix) {
						flash_drive.undefine();
						std::cout << "Deleted flash drive " << flash_drive.fd()->id() << std::endl;
						break;
					}
				}
			} catch (const std::exception& error) {
				std::cerr << "Couldn't remove flash drive " << fs::path(file).filename() << std::endl;
				std::cerr << error << std::endl;
			}
		}
	}

	//cleanup virtual machines
	for (auto& vm_folder: fs::directory_iterator(env->vm_metadata_dir())) {
		for (auto& file: fs::directory_iterator(vm_folder)) {
			try {
				if (fs::path(file).filename() == fs::path(vm_folder).filename()) {
					IR::Machine machine;
					machine.config = nlohmann::json::parse(get_metadata(file, "vm_config"));
					if (machine.vm()->prefix() == args.prefix) {
						machine.undefine();
						std::cout << "Deleted virtual machine " << machine.vm()->id() << std::endl;
						break;
					}
				}
			} catch (const std::exception& error) {
				std::cerr << "Couldn't remove virtual machine " << fs::path(file).filename() << std::endl;
				std::cerr << error << std::endl;
			}

		}
	}
	return 0;
}
