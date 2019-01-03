
#include "App.hpp"
#include <imgui.h>
#include <iostream>

App* app = nullptr;

App::App(): texture(640, 480) {
	::app = this;
	virtual_box = virtual_box_client.virtual_box();
}

void App::render() {
	if (ImGui::Begin("List of VMs")) {
		for (auto& machine: virtual_box.machines()) {
			bool is_selected = vm && (vm->machine().name() == machine.name());
			if (ImGui::Selectable(machine.name().c_str(), &is_selected)) {
				if (is_selected) {
					vm = std::make_shared<VM>(std::move(machine));
				} else {
					vm = nullptr;
				}
			}
		}
		ImGui::End();
	}

	if (vm && ImGui::Begin("VM"))  {
		std::lock_guard<std::mutex> lock_guard(mutex);
		ImGui::Image(texture.handle(), ImVec2(texture.width(), texture.height()));
		ImGui::End();
	}

	if (ImGui::Begin("FPS")) {
		ImGui::Text("%.1f", ImGui::GetIO().Framerate);
		ImGui::End();
	}
}
