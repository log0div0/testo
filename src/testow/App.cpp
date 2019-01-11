
#include "App.hpp"
#include <imgui.h>
#include <iostream>

App* app = nullptr;

App::App() {
	::app = this;
	virtual_box = virtual_box_client.virtual_box();
}

void App::render() {
	if (ImGui::Begin("List of VMs")) {
		for (auto& machine: virtual_box.machines()) {
			bool is_selected = vm && (vm->machine.name() == machine.name());
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
		std::shared_lock<std::shared_mutex> lock(vm->mutex);
		if (vm->screen) {
			if ((texture.width() != vm->screen->width()) || (texture.height() != vm->screen->height())) {
				texture = Texture(vm->screen->width(), vm->screen->height());
			}
			texture.write(vm->screen->data(), vm->screen->data_size());
			ImGui::Image(texture.handle(), ImVec2(texture.width(), texture.height()));
		} else {
			ImGui::Text("No signal");
		}
		ImGui::End();
	}

	if (ImGui::Begin("FPS")) {
		ImGui::Text("%.1f", ImGui::GetIO().Framerate);
		ImGui::End();
	}
}
