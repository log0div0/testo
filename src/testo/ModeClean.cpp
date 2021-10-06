
#include "backends/Environment.hpp"
#include "ModeClean.hpp"
#include "Utils.hpp"
#include "IR/Network.hpp"
#include "IR/FlashDrive.hpp"
#include "IR/Machine.hpp"
#include "Exceptions.hpp"
#include "Logger.hpp"

int clean_mode(const CleanModeArgs& args) {
	TRACE();

	//cleanup networks

	std::vector<IR::Network> networks_to_delete;
	std::vector<IR::FlashDrive> flash_drives_to_delete;
	std::vector<IR::Machine> machines_to_delete;

	if (fs::exists(env->network_metadata_dir())) {
		for (auto& network_folder: fs::directory_iterator(env->network_metadata_dir())) {
			for (auto& file: fs::directory_iterator(network_folder)) {
				try {
					if (fs::path(file).filename() == fs::path(network_folder).filename()) {
						IR::Network network;
						network.config = nlohmann::json::parse(get_metadata(file, "network_config"));
						if (network.nw()->prefix() == args.prefix) {
							networks_to_delete.push_back(network);
							break;
						}
					}
				} catch (const std::exception& error) {
					std::cerr << "Couldn't restore network controller from metadata: " << fs::path(file).filename() << std::endl;
					std::cerr << error << std::endl;
				}
			}
		}
	}

	//cleanup flash drives
	if (fs::exists(env->flash_drives_metadata_dir())) {
		for (auto& flash_drive_folder: fs::directory_iterator(env->flash_drives_metadata_dir())) {
			for (auto& file: fs::directory_iterator(flash_drive_folder)) {
				try {
					if (fs::path(file).filename() == fs::path(flash_drive_folder).filename()) {
						IR::FlashDrive flash_drive;
						flash_drive.config = nlohmann::json::parse(get_metadata(file, "fd_config"));
						if (flash_drive.fd()->prefix() == args.prefix) {
							flash_drives_to_delete.push_back(flash_drive);
							break;
						}
					}
				} catch (const std::exception& error) {
					std::cerr << "Couldn't restore flash drive controller from metadata: " << fs::path(file).filename() << std::endl;
					std::cerr << error << std::endl;
				}
			}
		}
	}


	//cleanup virtual machines
	if (fs::exists(env->vm_metadata_dir())) {
		for (auto& vm_folder: fs::directory_iterator(env->vm_metadata_dir())) {
			for (auto& file: fs::directory_iterator(vm_folder)) {
				try {
					if (fs::path(file).filename() == fs::path(vm_folder).filename()) {
						IR::Machine machine;
						machine.config = nlohmann::json::parse(get_metadata(file, "vm_config"));
						if (machine.vm()->prefix() == args.prefix) {
							machines_to_delete.push_back(machine);
							break;
						}
					}
				} catch (const std::exception& error) {
					std::cerr << "Couldn't restore virtual machine controller from metadata: " << fs::path(file).filename() << std::endl;
					std::cerr << error << std::endl;
				}
			}
		}
	}

	if (networks_to_delete.size() || flash_drives_to_delete.size() || machines_to_delete.size()) {
		if (!args.assume_yes) {
			std::cout << "Testo is about to erase the following entities:\n";
			std::cout << "Virtual networks:\n";
			for (auto& network: networks_to_delete) {
				std::cout << "\t- " << network.nw()->id() << std::endl;
			}
			std::cout << "Virtual flash drives:\n";
			for (auto& flash_drive: flash_drives_to_delete) {
				std::cout << "\t- " << flash_drive.fd()->id() << std::endl;
			}
			std::cout << "Virtual machines:\n";
			for (auto& machine: machines_to_delete) {
				std::cout << "\t- " << machine.vm()->id() << std::endl;
			}

			std::cout << "\nDo you confirm erasing these entities? [y/N]: ";
			std::string choice;
			std::getline(std::cin, choice);

			std::transform(choice.begin(), choice.end(), choice.begin(), ::toupper);

			if (choice != "Y" && choice != "YES") {
				throw std::runtime_error("Aborted");
			}
		}

		for (auto& network: networks_to_delete) {
			try {
				network.undefine();
				std::cout << "Deleted network " << network.nw()->id() << std::endl;
			} catch (const std::exception& error) {
				std::cerr << "Couldn't remove network " << network.nw()->id() << std::endl;
				std::cerr << error << std::endl;
			}
		}
		for (auto& flash_drive: flash_drives_to_delete) {
			try {
				flash_drive.undefine();
				std::cout << "Deleted flash drive " << flash_drive.fd()->id() << std::endl;
			} catch (const std::exception& error) {
				std::cerr << "Couldn't remove flash drive " << flash_drive.fd()->id() << std::endl;
				std::cerr << error << std::endl;
			}
		}
		for (auto& machine: machines_to_delete) {
			try {
				machine.undefine();
				std::cout << "Deleted virtual machine " << machine.vm()->id() << std::endl;
			} catch (const std::exception& error) {
				std::cerr << "Couldn't remove virtual machine " << machine.vm()->id() << std::endl;
				std::cerr << error << std::endl;
			}
		}

	}

	return 0;
}
