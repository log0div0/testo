
#include "App.hpp"
#include <imgui.h>
#include <iostream>
#include <darknet.h>

App* app = nullptr;

App::App():
	qemu_connect(vir::connect_open("qemu:///system"))
{
	::app = this;
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
					vm = std::make_unique<VM>(qemu_connect, domain);
				} else {
					vm = nullptr;
				}
			}
		}
		ImGui::End();
	}


	if (vm)  {
		std::shared_lock<std::shared_mutex> lock(vm->mutex);
		if (ImGui::Begin("VM")) {
			if (vm->view.width && vm->view.height) {
				if ((texture.width() != vm->view.width) ||
					(texture.height() != vm->view.height)) {
					texture = Texture(vm->view.width, vm->view.height);
					ImGui::SetWindowSize({texture.width() + 40, texture.height() + 40});
				}
				texture.write(vm->view.data(), vm->view.size());
				ImVec2 p = ImGui::GetCursorScreenPos();
				ImGui::GetWindowDrawList()->AddImage(texture.handle(), p, ImVec2(p.x + texture.width(), p.y + texture.height()));
			} else {
				ImGui::Text("No signal");
			}
			ImGui::End();
		}
		if (ImGui::Begin("Search params")) {
			vm->query = query;
			ImGui::InputText("query string", query, IM_ARRAYSIZE(query));
			ImGui::End();
		}
	}
}
