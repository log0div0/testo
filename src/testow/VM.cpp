
#include "VM.hpp"
#include "App.hpp"

using namespace std::chrono_literals;

VM::VM(vbox::Machine machine_): machine(std::move(machine_)), texture(640, 480) {
	session = ::app->virtual_box_client.session();
	machine.lock_machine(session, LockType_Shared);
	running = true;
	thread = std::thread([=] {
		run();
	});
}

VM::~VM() {
	running = false;
	thread.join();
	session.unlock_machine();
}

void VM::run() {
	auto interval = 80ms;
	auto previous = std::chrono::high_resolution_clock::now();
	while (running) {
		auto current = std::chrono::high_resolution_clock::now();
		auto diff = current - previous;
		if (diff < interval) {
			std::this_thread::sleep_for(interval - diff);
		}
		previous = current;

		if (!display.handle) {
			texture.clear();
			if (machine.state() == MachineState_Running) {
				try {
					display = session.console().display();
				} catch (const std::exception&) {
				}
			}
			continue;
		}

		ULONG width = 0;
		ULONG height = 0;
		ULONG bits_per_pixel = 0;
		LONG x_origin = 0;
		LONG y_origin = 0;
		GuestMonitorStatus guest_monitor_status = GuestMonitorStatus_Disabled;

		display.get_screen_resolution(0, &width, &height, &bits_per_pixel, &x_origin, &y_origin, &guest_monitor_status);

		if (!width || !height) {
			display = vbox::Display();
			continue;
		}
		if ((width != texture.width()) || (height != texture.height())) {
			std::lock_guard<std::shared_mutex> lock(mutex);
			texture = Texture(width, height);
		}

		vbox::SafeArray safe_array = display.take_screen_shot_to_array(0, width, height, BitmapFormat_BGRA);
		vbox::ArrayOut array_out = safe_array.copy_out(VT_UI1);

		texture.write(array_out.data, array_out.data_size);
	}
}
