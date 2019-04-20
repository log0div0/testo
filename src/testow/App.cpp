
#include "App.hpp"
#include <imgui.h>
#include <iostream>
#include <darknet.h>

App* app = nullptr;

App::App():
	// net("/home/alex/work/vbox/testo/nn/testo.cfg"),
	qemu_connect(vir::connect_open("qemu:///system"))
{
	::app = this;
	// net.load_weights("/home/alex/work/vbox/testo/nn/testo.weights");
	// net.set_batch(1);
	for (auto& domain: qemu_connect.domains({VIR_CONNECT_LIST_DOMAINS_PERSISTENT})) {
		domains.emplace(domain.name(), std::move(domain));
	}
}

void App::render() {
	if (ImGui::Begin("List of VMs")) {
		for (auto& [domain_name, domain]: domains) {
			bool is_selected = vm && (vm->domain_name == domain_name);
			if (ImGui::Selectable(domain_name.c_str(), &is_selected)) {
				if (is_selected) {
					vm = nullptr;
					vm = std::make_shared<VM>(qemu_connect, domain);
				} else {
					vm = nullptr;
				}
			}
		}
		ImGui::End();
	}

	if (vm && ImGui::Begin("VM"))  {
		ImVec2 p = ImGui::GetCursorScreenPos();
		std::shared_lock<std::shared_mutex> lock(vm->mutex);
		if (vm->width && vm->height) {
			if ((width != vm->width) || (height != vm->height)) {
				width = vm->width;
				height = vm->height;
				texture1 = Texture(width, height);
				texture2 = Texture(width, height);
			}
			texture1.write(vm->texture1.data(), vm->texture1.size());
			// texture2.write(vm->texture2.data(), vm->texture2.size());
			ImGui::GetWindowDrawList()->AddImage(texture1.handle(), p, ImVec2(p.x + width, p.y + height));
			// ImGui::GetWindowDrawList()->AddImage(texture2.handle(), p, ImVec2(p.x + width, p.y + height));
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
