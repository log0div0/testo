
#include "App.hpp"
#include <imgui.h>
#include <iostream>

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

App* app = nullptr;

App::App(std::shared_ptr<Hypervisor> hypervisor)
{
	::app = this;
	guests = hypervisor->guests();
}

void App::render() {
	if (ImGui::Begin("List of VMs")) {
		for (auto& guest: guests) {
			bool is_selected = vm && (vm->guest == guest);
			if (ImGui::Selectable(guest->name().c_str(), &is_selected)) {
				if (is_selected) {
					vm = nullptr;
					vm = std::make_unique<VM>(guest);
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
					ImGui::SetWindowSize({
						float(texture.width() + 40),
						float(texture.height() + 40)
					});
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
