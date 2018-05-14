
#include <iostream>
#include <regex>
#include <chrono>
#include <thread>
#include "vbox/api.hpp"
#include "vbox/virtual_box_client.hpp"
#include "vbox/virtual_box.hpp"
#include "vbox/unlocker.hpp"
#include "vbox/virtual_box_error_info.hpp"
#include <SDL2/SDL.h>

using namespace std::chrono_literals;

void backtrace(std::ostream& stream, const std::exception& error, size_t n) {
	stream << n << ". " << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << std::endl;
		backtrace(stream, error, n + 1);
	} catch(...) {
		stream << std::endl;
		stream << n << ". " << "[Unknown exception type]";
	}
}

std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error, 1);
	return stream;
}

int main(int argc, char* argv[]) {
	try {
		vbox::API api;
		vbox::VirtualBoxClient virtual_box_client;
		vbox::VirtualBox virtual_box = virtual_box_client.virtual_box();
		vbox::Session session = virtual_box_client.session();

		std::vector<vbox::Machine> machines = virtual_box.machines();
		for (auto& machine: machines) {
			if (machine.name() == "ubuntu_2") {
				if (machine.session_state() != SessionState_Unlocked) {
					machine.lock_machine(session, LockType_Shared);
					vbox::Unlocker unlocker(session);
					session.console().power_down().wait_and_throw_if_failed();
				}
				while (machine.session_state() != SessionState_Unlocked) {
					std::this_thread::sleep_for(100ms);
				}
				machine.delete_config(machine.unregister(CleanupMode_DetachAllReturnHardDisksOnly)).wait_and_throw_if_failed();
			}
		}

		vbox::Machine machine = virtual_box.find_machine("ubuntu");
		std::cout << machine << std::endl;

		vbox::GuestOSType guest_os_type = virtual_box.get_guest_os_type("ubuntu_64");

		std::string settings_file_path = virtual_box.compose_machine_filename("ubuntu_2", "/", {}, {});
		std::cout << settings_file_path << std::endl;
		machine = virtual_box.create_machine(settings_file_path, "ubuntu_2", {"/"}, guest_os_type.id(), {});
		machine.memory_size(guest_os_type.recommended_ram());
		machine.vram_size(guest_os_type.recommended_vram());
		machine.save_settings();
		virtual_box.register_machine(machine);
		{
			machine.lock_machine(session, LockType_Write);
			vbox::Unlocker unlocker(session);

			vbox::Machine machine = session.machine();

			vbox::StorageController ide = machine.add_storage_controller("IDE", StorageBus_IDE);
			vbox::StorageController sata = machine.add_storage_controller("SATA", StorageBus_SATA);
			ide.port_count(2);
			sata.port_count(1);

			vbox::Medium dvd = virtual_box.open_medium("C:\\Users\\log0div0\\Downloads\\ubuntu-16.04.4-server-amd64.iso",
				DeviceType_DVD, AccessMode_ReadOnly, false);
			machine.attach_device(ide.name(), 1, 0, DeviceType_DVD, dvd);

			std::string hard_disk_path = std::regex_replace(settings_file_path, std::regex("\\.vbox$"), ".vdi");
			std::cout << hard_disk_path << std::endl;
			vbox::Medium hard_disk = virtual_box.create_medium("vdi", hard_disk_path, AccessMode_ReadWrite, DeviceType_HardDisk);
			hard_disk.create_base_storage(8ull * 1024 * 1024 * 1024, MediumVariant_Standard).wait_and_throw_if_failed();
			machine.attach_device(sata.name(), 0, 0, DeviceType_HardDisk, hard_disk);
			machine.save_settings();
		}
		std::cout << machine << std::endl;
		/*
		{
			machine.launch_vm_process(session, "headless").wait_and_throw_if_failed();
			vbox::Unlocker unlocker(session);
			vbox::Console console = session.console();
			vbox::Display display = console.display();

			vbox::Framebuffer framebuffer(new vbox::IFramebuffer);
			display.attach_framebuffer(0, framebuffer);

			for (size_t i = 0; i < 2000; ++i) {
				vbox::api->pfnProcessEventQueue(100);
			}

			console.power_down().wait_and_throw_if_failed();
		}
		*/
		if (SDL_Init(SDL_INIT_VIDEO) < 0) {
			fprintf(stderr, "could not initialize sdl2: %s\n", SDL_GetError());
			return 1;
		}
		SDL_Window* window = SDL_CreateWindow(
			"hello_sdl2",
			SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
			600, 400,
			SDL_WINDOW_SHOWN
		);
		if (window == NULL) {
			fprintf(stderr, "could not create window: %s\n", SDL_GetError());
			return 1;
		}
		SDL_Surface* screenSurface = SDL_GetWindowSurface(window);
		SDL_FillRect(screenSurface, NULL, SDL_MapRGB(screenSurface->format, 0xFF, 0xFF, 0xFF));
		SDL_UpdateWindowSurface(window);
		SDL_Delay(2000);
		SDL_DestroyWindow(window);
		SDL_Quit();
		return 0;
	} catch (const std::exception& error) {
		std::cout << error << std::endl;
		return 1;
	}
}
