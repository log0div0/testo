
#include "App.hpp"
#include <imgui.h>
#include <iostream>
#include <darknet.h>

App* app = nullptr;

App::App(): net("C:\\Users\\log0div0\\work\\testo\\nn\\testo.cfg") {
	::app = this;
	net.load_weights("C:\\Users\\log0div0\\work\\testo\\nn\\testo.weights");
	virtual_box = virtual_box_client.virtual_box();
}

void App::render() {
	if (ImGui::Begin("List of VMs")) {
		for (auto& machine: virtual_box.machines()) {
			bool is_selected = vm && (vm->machine.name() == machine.name());
			if (ImGui::Selectable(machine.name().c_str(), &is_selected)) {
				if (is_selected) {
					vm = nullptr;
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
		if (vm->width && vm->height) {
			if ((width != vm->width) || (height != vm->height)) {
				width = vm->width;
				height = vm->height;
				texture1 = Texture(width, height);
				texture2 = Texture(width, height);
			}
			texture1.write(vm->texture1.data(), vm->texture1.size());
			texture2.write(vm->texture2.data(), vm->texture2.size());
			ImVec2 p = ImGui::GetCursorScreenPos();
			ImGui::GetWindowDrawList()->AddImage(texture1.handle(), p, ImVec2(p.x + width, p.y + height));
			ImGui::GetWindowDrawList()->AddImage(texture2.handle(), p, ImVec2(p.x + width, p.y + height));
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
