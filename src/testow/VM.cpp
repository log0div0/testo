
#include "VM.hpp"
#include "App.hpp"

using namespace std::chrono_literals;

void Screen::update() {
	ULONG width = 0;
	ULONG height = 0;
	ULONG bits_per_pixel = 0;
	LONG x_origin = 0;
	LONG y_origin = 0;
	GuestMonitorStatus guest_monitor_status = GuestMonitorStatus_Disabled;

	_display.get_screen_resolution(0, &width, &height, &bits_per_pixel, &x_origin, &y_origin, &guest_monitor_status);

	if (!width || !height) {
		return;
	}

	_width = width;
	_height = height;

	vbox::SafeArray safe_array = _display.take_screen_shot_to_array(0, width, height, BitmapFormat_BGRA);
	_pixels = safe_array.copy_out(VT_UI1);
}

VM::VM(vbox::Machine machine_): machine(std::move(machine_)) {
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
	auto interval = 200ms;
	auto previous = std::chrono::high_resolution_clock::now();
	while (running) {
		auto current = std::chrono::high_resolution_clock::now();
		auto diff = current - previous;
		if (diff < interval) {
			std::this_thread::sleep_for(interval - diff);
		}
		previous = current;

		std::lock_guard<std::shared_mutex> lock(mutex);
		if (machine.state() == MachineState_Running) {
			if (!screen) {
				try {
					screen = std::make_unique<Screen>(session.console().display());
				} catch (const std::exception&) {
					continue;
				}
			}
			screen->update();
		} else {
			screen = nullptr;
		}
	}
}
