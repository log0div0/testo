
#include "Vbox.hpp"

VboxGuest::VboxGuest(vbox::Machine machine_, vbox::Session session_): Guest(machine_.name()),
	machine(std::move(machine_)), session(std::move(session_))
{
	machine.lock_machine(session, LockType_Shared);
}

VboxGuest::~VboxGuest() {
	session.unlock_machine();
}

stb::Image VboxGuest::screenshot() {
	if (machine.state() != MachineState_Running) {
		return {};
	}

	auto display = session.console().display();

	ULONG width = 0;
	ULONG height = 0;
	ULONG bits_per_pixel = 0;
	LONG x_origin = 0;
	LONG y_origin = 0;
	GuestMonitorStatus guest_monitor_status = GuestMonitorStatus_Disabled;

	display.get_screen_resolution(0, &width, &height, &bits_per_pixel, &x_origin, &y_origin, &guest_monitor_status);

	if (!width || !height) {
		return {};
	}

	stb::Image result(width, height, 3);

	vbox::SafeArray safe_array = display.take_screen_shot_to_array(0, width, height, BitmapFormat_BGRA);
	vbox::ArrayOut array_out = safe_array.copy_out(VT_UI1);

	for(size_t h = 0; h < height; ++h){
		for(size_t w = 0; w < width; ++w){
			for(size_t c = 0; c < 3; ++c){
				size_t src_index = h*width*4 + w*4 + c;
				size_t dst_index = h*width*3 + w*3 + c;
				result._data[dst_index] = array_out[src_index];
			}
		}
	}

	return result;
}

Vbox::Vbox() {
	virtual_box = virtual_box_client.virtual_box();
}

std::vector<std::shared_ptr<Guest>> Vbox::guests() const {
	std::vector<std::shared_ptr<Guest>> result;
	for (auto& machine: virtual_box.machines()) {
		result.push_back(std::make_shared<VboxGuest>(std::move(machine), virtual_box_client.session()));
	}
	return result;
}