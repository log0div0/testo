
#include "App.hpp"
#include <imgui.h>
#include <iostream>
#include <sstream>
#include <clipp.h>

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

App::App(int argc, char** argv)
{
	using namespace clipp;

	std::string hypervisor_name;

	auto cli = (
		option("-v", "--hypervisor") & value("hypervisor name", hypervisor_name)
	);

	if (!parse(argc, argv, cli)) {
		std::stringstream ss;
		ss << make_man_page(cli, argv[0]);
		throw std::runtime_error(ss.str());
	}

	hypervisor = Hypervisor::get(hypervisor_name);

	::app = this;
	guests = hypervisor->guests();
}

const char* colors[] = {
	"white",
	"gray",
	"black",
	"red",
	"orange",
	"yellow",
	"green",
	"cyan",
	"blue",
	"purple",
};

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
				texture.write(vm->view.data, vm->view.size());
				ImVec2 p = ImGui::GetCursorScreenPos();
				ImGui::GetWindowDrawList()->AddImage(texture.handle(), p, ImVec2(p.x + texture.width(), p.y + texture.height()));
			} else {
				ImGui::Text("No signal");
			}
			ImGui::End();
		}
		if (ImGui::Begin("Search params")) {
			vm->query = query;
			if (foreground >= 0) {
				vm->foreground = colors[foreground];
			}
			if (background >= 0) {
				vm->background = colors[background];
			}
			ImGui::InputText("query string", query, IM_ARRAYSIZE(query));
			ImGui::ListBox("foreground", &foreground, colors, IM_ARRAYSIZE(colors));
			ImGui::ListBox("background", &background, colors, IM_ARRAYSIZE(colors));
			ImGui::End();
		}
	}
}
