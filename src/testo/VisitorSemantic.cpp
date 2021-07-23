
#include "VisitorSemantic.hpp"
#include <nn/Homm3Object.hpp>
#include "backends/Environment.hpp"
#include "Exceptions.hpp"
#include "IR/Program.hpp"
#include "Parser.hpp"
#include <fmt/format.h>
#include <wildcards.hpp>

void VisitorSemanticConfig::validate() const {

}

void VisitorSemanticConfig::dump(nlohmann::json& j) const {
	j["prefix"] = prefix;
}

VisitorSemantic::VisitorSemantic(const VisitorSemanticConfig& config) {
	prefix = config.prefix;

	keys = {
		"ESC",
		"ONE",
		"TWO",
		"THREE",
		"FOUR",
		"FIVE",
		"SIX",
		"SEVEN",
		"EIGHT",
		"NINE",
		"ZERO",
		"MINUS",
		"EQUALSIGN",
		"BACKSPACE",
		"TAB",
		"Q",
		"W",
		"E",
		"R",
		"T",
		"Y",
		"U",
		"I",
		"O",
		"P",
		"LEFTBRACE",
		"RIGHTBRACE",
		"ENTER",
		"LEFTCTRL",
		"A",
		"S",
		"D",
		"F",
		"G",
		"H",
		"J",
		"K",
		"L",
		"SEMICOLON",
		"APOSTROPHE",
		"GRAVE",
		"LEFTSHIFT",
		"BACKSLASH",
		"Z",
		"X",
		"C",
		"V",
		"B",
		"N",
		"M",
		"COMMA",
		"DOT",
		"SLASH",
		"RIGHTSHIFT",
		"LEFTALT",
		"SPACE",
		"CAPSLOCK",
		"F1",
		"F2",
		"F3",
		"F4",
		"F5",
		"F6",
		"F7",
		"F8",
		"F9",
		"F10",
		"F11",
		"F12",
		"NUMLOCK",
		"KP_0",
		"KP_1",
		"KP_2",
		"KP_3",
		"KP_4",
		"KP_5",
		"KP_6",
		"KP_7",
		"KP_8",
		"KP_9",
		"KP_PLUS",
		"KP_MINUS",
		"KP_SLASH",
		"KP_ASTERISK",
		"KP_ENTER",
		"KP_DOT",
		"SCROLLLOCK",
		"RIGHTCTRL",
		"RIGHTALT",
		"HOME",
		"UP",
		"PAGEUP",
		"LEFT",
		"RIGHT",
		"END",
		"DOWN",
		"PAGEDOWN",
		"INSERT",
		"DELETE",
		"SCROLLUP",
		"SCROLLDOWN",
		"LEFTMETA",
		"RIGHTMETA",
	};

	attr_ctxs.insert({"vm_global", {
		{"ram", {false, Token::category::size}},
		{"iso", {false, Token::category::quoted_string}},
		{"nic", {true, Token::category::attr_block}},
		{"disk", {true, Token::category::attr_block}},
		{"video", {true, Token::category::attr_block}},
		{"shared_folder", {true, Token::category::attr_block}},
		{"cpus", {false, Token::category::number}},
		{"qemu_spice_agent", {false, Token::category::boolean}},
		{"qemu_enable_usb3", {false, Token::category::boolean}},
		{"loader", {false, Token::category::quoted_string}},
	}});

	attr_ctxs.insert({"disk", {
		{"size", {false, Token::category::size}},
		{"source", {false, Token::category::quoted_string}},
	}});

	attr_ctxs.insert({"nic", {
		{"attached_to", {false, Token::category::quoted_string}},
		{"attached_to_dev", {false, Token::category::quoted_string}},
		{"mac", {false, Token::category::quoted_string}},
		{"adapter_type", {false, Token::category::quoted_string}},
	}});

	attr_ctxs.insert({"video", {
		{"qemu_mode", {false, Token::category::quoted_string}}, // deprecated
		{"adapter_type", {false, Token::category::quoted_string}},
	}});

	attr_ctxs.insert({"shared_folder", {
		{"host_path", {false, Token::category::quoted_string}},
		{"readonly", {false, Token::category::boolean}},
	}});

	attr_ctxs.insert({"fd_global", {
		{"fs", {false, Token::category::quoted_string}},
		{"size", {false, Token::category::size}},
		{"folder", {false, Token::category::quoted_string}},
	}});

	attr_ctxs.insert({"network_global", {
		{"mode", {false, Token::category::quoted_string}},
	}});

	attr_ctxs.insert({"test_global", {
		{"no_snapshots", {false, Token::category::boolean}},
		{"description", {false, Token::category::quoted_string}},
	}});
}

static uint32_t size_to_mb(const std::string& size) {
	uint32_t result = std::stoul(size.substr(0, size.length() - 2));
	if (size[size.length() - 2] == 'M') {
		result = result * 1;
	} else if (size[size.length() - 2] == 'G') {
		result = result * 1024;
	} else {
		throw Exception("Unknown size specifier"); //should not happen ever
	}

	return result;
}

static bool str_to_bool(const std::string& str) {
	if (str == "true") {
		return true;
	} else if (str == "false") {
		return false;
	} else {
		throw std::runtime_error("Can't convert \"" + str + "\" to boolean");
	}
}

void VisitorSemantic::visit() {
	for (auto& test: IR::program->all_selected_tests) {
		visit_test(test);
	}
	for (auto& test: IR::program->all_selected_tests) {
		//Now that we've checked that all commands are ligit we could check that
		//all parents have totally separate vms. We can't do that before command block because
		//a user may specify unexisting vmc in some command and we need to catch that before that hierarchy check

		std::vector<std::set<std::shared_ptr<IR::Machine>>> parents_subtries_vm;
		std::vector<std::set<std::shared_ptr<IR::FlashDrive>>> parents_subtries_fd;

		//populate our parents paths
		for (auto parent: test->parents) {
			parents_subtries_vm.push_back(parent->get_all_machines());
			parents_subtries_fd.push_back(parent->get_all_flash_drives());
		}

		//check that parents path are independent
		for (size_t i = 0; i < parents_subtries_vm.size(); ++i) {
			for (size_t j = 0; j < parents_subtries_vm.size(); ++j) {
				if (i == j) {
					continue;
				}

				std::vector<std::shared_ptr<IR::Machine>> intersection;

				std::set_intersection(
					parents_subtries_vm[i].begin(), parents_subtries_vm[i].end(),
					parents_subtries_vm[j].begin(), parents_subtries_vm[j].end(),
					std::back_inserter(intersection));

				if (intersection.size() != 0) {
					throw Exception(std::string(test->ast_node->begin()) + ": Error: some parents have common virtual machines");
				}
			}
		}

		//check that parents path are independent
		for (size_t i = 0; i < parents_subtries_fd.size(); ++i) {
			for (size_t j = 0; j < parents_subtries_fd.size(); ++j) {
				if (i == j) {
					continue;
				}

				std::vector<std::shared_ptr<IR::FlashDrive>> intersection;

				std::set_intersection(
					parents_subtries_fd[i].begin(), parents_subtries_fd[i].end(),
					parents_subtries_fd[j].begin(), parents_subtries_fd[j].end(),
					std::back_inserter(intersection));

				if (intersection.size() != 0) {
					throw Exception(std::string(test->ast_node->begin()) + ": Error: some parents have common flash drives");
				}
			}
		}
	}
}

void VisitorSemantic::visit_macro(std::shared_ptr<IR::Macro> macro) {
	auto result = visited_macros.insert(macro);
	if (!result.second) {
		return;
	}

	macro->validate();
}

void VisitorSemantic::visit_test(std::shared_ptr<IR::Test> test) {
	try {
		StackPusher<VisitorSemantic> new_ctx(this, test->stack);

		if (test->ast_node->attrs) {
			test->attrs = visit_attr_block(test->ast_node->attrs, "test_global");
		}

		current_test = test;

		current_test->cksum_input << "TEST NAME = " << test->name() << std::endl;
		current_test->cksum_input << "PARENTS IN ALPHABETICAL ORDER = ";
		std::vector<std::string> parents_names;
		for (auto parent: test->parents) {
			parents_names.push_back(parent->name());
		}
		std::sort(parents_names.begin(), parents_names.end());
		for (size_t i = 0; i < parents_names.size(); ++i) {
			if (i) {
				current_test->cksum_input << " ";
			}
			current_test->cksum_input << parents_names.at(i);
		}
		current_test->cksum_input << std::endl;
		current_test->cksum_input << "SNAPSHOT NEEDED = " << test->snapshots_needed() << std::endl;

		visit_command_block(test->ast_node->cmd_block);

		std::hash<std::string> h;
		current_test->cksum = std::to_string(h(current_test->cksum_input.str()));

		current_test = nullptr;
	} catch (const ControllerCreatonException& error) {
		throw;
	} catch (const Exception& error) {
		if (test->macro_call_stack.size()) {
			std::stringstream ss;
			for (auto macro_call: test->macro_call_stack) {
				ss << std::string(macro_call->begin()) + std::string(": In a macro call ") << macro_call->name().value() << std::endl;
			}

			std::string msg = ss.str();
			std::throw_with_nested(Exception(msg.substr(0, msg.length() - 1)));
		} else {
			throw;
		}
	}
}

void VisitorSemantic::visit_command_block(std::shared_ptr<AST::CmdBlock> block) {
	for (auto command: block->commands) {
		visit_command(command);
	}
}

void VisitorSemantic::visit_command(std::shared_ptr<AST::ICmd> cmd) {
	if (auto p = std::dynamic_pointer_cast<AST::Cmd<AST::RegularCmd>>(cmd)) {
		visit_regular_command({p->cmd, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Cmd<AST::MacroCall>>(cmd)) {
		visit_cmd_macro_call({p->cmd, stack});
	} else {
		throw Exception("Should never happen");
	}
}

void VisitorSemantic::visit_regular_command(const IR::RegularCommand& regular_cmd) {
	current_test->cksum_input << regular_cmd.entity() << " {" << std::endl;
	if ((current_controller = IR::program->get_machine_or_null(regular_cmd.entity()))) {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		visit_machine(vmc);

		if (vmc->config.count("nic")) {
			auto& nics = vmc->config.at("nic");
			for (auto& nic: nics) {
				if (nic.count("attached_to")) {
					std::string network_name = nic.at("attached_to");
					auto network = IR::program->get_network_or_null(network_name);
					if (!network) {
						try {
							throw std::runtime_error(fmt::format("NIC \"{}\" is attached to an unknown network: \"{}\"",
								nic.at("name").get<std::string>(), network_name));
						} catch (const std::exception& error) {
							std::throw_with_nested(ControllerCreatonException(vmc));
						}
					}
					visit_network(network);
					nic["network_mode"] = network->config.at("mode");
				}
			}
		}

		visit_action_vm(regular_cmd.ast_node->action);
	} else if ((current_controller = IR::program->get_flash_drive_or_null(regular_cmd.entity()))) {
		auto fdc = std::dynamic_pointer_cast<IR::FlashDrive>(current_controller);
		visit_flash(fdc);
		visit_action_fd(regular_cmd.ast_node->action);
	} else {
		throw Exception(std::string(regular_cmd.ast_node->entity->begin()) + ": Error: unknown virtual entity: " + regular_cmd.entity());
	}
	current_test->cksum_input << "}" << std::endl;

	current_controller = nullptr;
}

void VisitorSemantic::visit_action_block(std::shared_ptr<AST::ActionBlock> action_block) {
	for (auto action: action_block->actions) {
		visit_action(action);
	}
}

void VisitorSemantic::visit_action(std::shared_ptr<AST::IAction> action) {
	if (std::dynamic_pointer_cast<IR::Machine>(current_controller)) {
		visit_action_vm(action);
	} else {
		visit_action_fd(action);
	}
}

void VisitorSemantic::visit_action_vm(std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Abort>>(action)) {
		visit_abort({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		visit_print({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Type>>(action)) {
		visit_type({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Press>>(action)) {
		visit_press({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Hold>>(action)) {
		visit_hold({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Release>>(action)) {
		visit_release({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Mouse>>(action)) {
		visit_mouse({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Plug>>(action)) {
		visit_plug({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Start>>(action)) {
		visit_start({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Stop>>(action)) {
		visit_stop({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Shutdown>>(action)) {
		visit_shutdown({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Exec>>(action)) {
		visit_exec({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		visit_copy({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Screenshot>>(action)) {
		visit_screenshot({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Wait>>(action)) {
		visit_wait({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		visit_sleep({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroCall>>(action)) {
		visit_action_macro_call({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		visit_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		visit_for_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		visit_cycle_control({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
		// do nothing
	} else {
		throw Exception(std::string(action->begin()) + ": Error: The action \"" + action->t.value() + "\" is not applicable to a virtual machine");
	}
}

void VisitorSemantic::visit_action_fd(std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Abort>>(action)) {
		visit_abort({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		visit_print({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		visit_copy({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		visit_sleep({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
		// do nothing
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroCall>>(action)) {
		visit_action_macro_call({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		visit_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		visit_for_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		visit_cycle_control({p->action, stack});
	} else {
		throw Exception(std::string(action->begin()) + ": Error: The action \"" + action->t.value() + "\" is not applicable to a flash drive");
	}
}

void VisitorSemantic::visit_abort(const IR::Abort& abort) {
	current_test->cksum_input << "abort \"" << abort.message() << "\"" << std::endl;
}

void VisitorSemantic::visit_print(const IR::Print& print) {
	current_test->cksum_input << "print \"" << print.message() << "\"" << std::endl;
}

void VisitorSemantic::visit_type(const IR::Type& type) {
	current_test->cksum_input << "type "
		<< "\"" << type.text() << "\""
		<< " interval " << type.interval() << std::endl;
}

void VisitorSemantic::visit_press(const IR::Press& press) {
	current_test->cksum_input << "press ";

	int i = 0;
	for (auto key_spec: press.ast_node->keys) {
		if (i++) {
			current_test->cksum_input << ",";
		}
		visit_key_spec({key_spec, stack});
	}

	current_test->cksum_input << " interval " << press.interval() << std::endl;
}

void VisitorSemantic::visit_key_combination(std::shared_ptr<AST::KeyCombination> combination) {
	for (size_t i = 0; i < combination->buttons.size(); ++i) {
		auto button = combination->buttons[i];
		if (!is_button(button)) {
			throw Exception(std::string(button.begin()) +
				" :Error: unknown key: " + button.value());
		}

		for (size_t j = i + 1; j < combination->buttons.size(); ++j) {
			if (button.value() == combination->buttons[j].value()) {
				throw Exception(std::string(combination->buttons[j].begin()) +
					" :Error: duplicate key: " + button.value());
			}
		}

		if (i) {
			current_test->cksum_input << "+";
		}
		std::string button_str = button.value();
		std::transform(button_str.begin(), button_str.end(), button_str.begin(), ::toupper);
		current_test->cksum_input << button_str;
	}
}

void VisitorSemantic::visit_key_spec(const IR::KeySpec& key_spec) {
	visit_key_combination(key_spec.ast_node->combination);

	auto times = key_spec.times();

	if (times < 1) {
		throw Exception(std::string(key_spec.ast_node->times->begin()) +
			" :Error: can't press a button less than 1 time: " + std::to_string(times));
	}

	current_test->cksum_input << "*" << times;
}

void VisitorSemantic::visit_hold(const IR::Hold& hold) {
	current_test->cksum_input << "hold ";
	visit_key_combination(hold.ast_node->combination);
	current_test->cksum_input << std::endl;
}

void VisitorSemantic::visit_release(const IR::Release& release) {
	current_test->cksum_input << "release";
	if (release.ast_node->combination) {
		current_test->cksum_input << " ";
		visit_key_combination(release.ast_node->combination);
	}
	current_test->cksum_input << std::endl;
}

void VisitorSemantic::visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers)
{
	//finally we're here

	/*
	what checks do we need?
	1) If we have from, there could not be another from
	2) If we have center, there could not be another center
	3) From could not be after center or move
	4) Center could not be after move
	This should cover it
	*/

	bool has_from = false;
	bool has_center = false;
	bool has_move = false;

	for (auto specifier: specifiers) {
		auto arg = specifier->arg;

		current_test->cksum_input << std::string(*specifier);
		if (specifier->is_from()) {
			if (!arg) {
				throw Exception(std::string(specifier->begin()) + ": Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}

			auto i = std::stoi(arg.value());
			if (i < 0) {
				throw Exception(std::string(arg.begin()) + ": Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}

			if (has_from) {
				throw Exception(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after another \"from\" specifier");
			}
			if (has_center) {
				throw Exception(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after a \"precision\" specifier");
			}
			if (has_move) {
				throw Exception(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after a \"move\" specifier");
			}
			has_from = true;
			continue;
		} if (specifier->is_centering()) {
			if (arg) {
				throw Exception(std::string(specifier->begin()) + ": Error: specifier " + specifier->name.value() + " must not have an argument");
			}
			if (has_center) {
				throw Exception(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after another \"precision\" specifier");
			}
			if (has_move) {
				throw Exception(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after a \"move\" specifier");
			}
			has_center = true;
			continue;
		} else if (specifier->is_moving()) {
			if (!arg) {
				throw Exception(std::string(specifier->begin()) + ": Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}

			auto i = std::stoi(arg.value());
			if (i < 0) {
				throw Exception(std::string(arg.begin()) + ": Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}
			has_move = true;
			continue;
		} else {
			throw Exception(std::string(specifier->begin()) + ": Error: unknown specifier: " + specifier->name.value());
		}

	}
}

void VisitorSemantic::visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates) {
	if (coordinates.x_is_relative() ^ coordinates.y_is_relative()) {
		throw std::runtime_error(std::string(coordinates.ast_node->begin()) + ": Error: mouse coordinates must be either both absolute either both relative");
	}

	current_test->cksum_input << coordinates.x() << " " << coordinates.y();
}

void VisitorSemantic::visit_select_js(const IR::SelectJS& js) {
	auto script = js.script();

	if (!script.length()) {
		throw Exception(std::string(js.ast_node->begin()) + ": Error: empty script in js selection");
	}

	/*try {
		validate_js(script);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(std::string(js.ast_node->begin()) + ": Error while validating js selection"));
	}*/

	current_test->cksum_input << "js \"" << script << "\"";
}

void VisitorSemantic::visit_select_img(const IR::SelectImg& img) {
	auto img_path = img.img_path();

	if (!fs::exists(img_path)) {
		throw Exception(std::string(img.ast_node->begin()) + ": Error: specified image path does not exist: " + img_path.generic_string());
	}

	if (!fs::is_regular_file(img_path)) {
		throw Exception(std::string(img.ast_node->begin()) + ": Error: specified image path does not lead to a regular file: " + img_path.generic_string());
	}

	current_test->cksum_input
		<< "img \"" << img_path.generic_string() << "\""
		<< " (file signature = " << file_signature(img_path) << ")";
}

void VisitorSemantic::visit_select_homm3(const IR::SelectHomm3& homm3) {
	auto id = homm3.id();

	if (!nn::Homm3Object::check_class_name(id)) {
		throw Exception(std::string(homm3.ast_node->begin()) + ": Error: specified Heroes of Might and Magic object does not exist " + id);
	}

	current_test->cksum_input << "homm3 \"" << id << "\"";
}

void VisitorSemantic::visit_select_text(const IR::SelectText& text) {
	auto txt = text.text();
	if (!txt.length()) {
		throw Exception(std::string(text.ast_node->begin()) + ": Error: empty string in text selection");
	}

	current_test->cksum_input << "text \"" << txt << "\"";
}

void VisitorSemantic::visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable) {
	if (mouse_selectable.ast_node->selectable->is_negated()) {
		throw Exception(std::string(mouse_selectable.ast_node->begin()) + ": Error: negation is not supported for mouse move/click actions");
	}

	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(mouse_selectable.ast_node->selectable)) {
		if (mouse_selectable.ast_node->specifiers.size()) {
			throw Exception(std::string(mouse_selectable.ast_node->specifiers[0]->begin()) + ": Error: mouse specifiers are not supported for js selections");
		}
		visit_select_js({p->selectable, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectText>>(mouse_selectable.ast_node->selectable)) {
		visit_select_text({p->selectable, stack});
		visit_mouse_additional_specifiers(mouse_selectable.ast_node->specifiers);
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectImg>>(mouse_selectable.ast_node->selectable)) {
		visit_select_img({p->selectable, stack});
		visit_mouse_additional_specifiers(mouse_selectable.ast_node->specifiers);
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectHomm3>>(mouse_selectable.ast_node->selectable)) {
		visit_select_homm3({p->selectable, stack});
		visit_mouse_additional_specifiers(mouse_selectable.ast_node->specifiers);
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectParentedExpr>>(mouse_selectable.ast_node->selectable)) {
		throw Exception(std::string(mouse_selectable.ast_node->begin()) + ": Error: select expressions are not supported for mouse move/click actions");
	}
}

void VisitorSemantic::visit_mouse_move_click(const IR::MouseMoveClick& mouse_move_click) {
	current_test->cksum_input << mouse_move_click.event_type();
	if (mouse_move_click.ast_node->object) {
		current_test->cksum_input << " ";
		if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseCoordinates>>(mouse_move_click.ast_node->object)) {
			visit_mouse_move_coordinates({p->target, stack});
		} else if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseSelectable>>(mouse_move_click.ast_node->object)) {
			visit_mouse_move_selectable({p->target, stack});
		}
	}
}

void VisitorSemantic::visit_mouse_hold(const IR::MouseHold& mouse_hold) {
	current_test->cksum_input << "hold " << mouse_hold.button() << std::endl;
}

void VisitorSemantic::visit_mouse_release(const IR::MouseRelease& mouse_release) {
	current_test->cksum_input << "release" << std::endl;
}

void VisitorSemantic::visit_mouse(const IR::Mouse& mouse) {
	current_test->cksum_input << "mouse ";

	if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseMoveClick>>(mouse.ast_node->event)) {
		visit_mouse_move_click({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseHold>>(mouse.ast_node->event)) {
		visit_mouse_hold({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseRelease>>(mouse.ast_node->event)) {
		visit_mouse_release({p->event, stack});
	}

	current_test->cksum_input << std::endl;
}

void VisitorSemantic::visit_plug(const IR::Plug& plug) {
	if (plug.is_on()) {
		current_test->cksum_input << "plug ";
	} else {
		current_test->cksum_input << "unplug ";
	}

	if (auto p = std::dynamic_pointer_cast<AST::PlugResource<AST::PlugFlash>>(plug.ast_node->resource)) {
		visit_plug_flash({p->resource, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::PlugResource<AST::PlugDVD>>(plug.ast_node->resource)) {
		visit_plug_dvd({p->resource, stack}, plug.is_on());
	} else if (auto p = std::dynamic_pointer_cast<AST::PlugResource<AST::PlugNIC>>(plug.ast_node->resource)) {
		visit_plug_nic({p->resource, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::PlugResource<AST::PlugLink>>(plug.ast_node->resource)) {
		visit_plug_link({p->resource, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::PlugResource<AST::PlugHostDev>>(plug.ast_node->resource)) {
		visit_plug_hostdev({p->resource, stack});
	} else {
		throw Exception(std::string("unknown hardware type to plug/unplug: ") +
			plug.ast_node->resource->t.value());
	}

	current_test->cksum_input << std::endl;
}

void VisitorSemantic::visit_plug_flash(const IR::PlugFlash& plug_flash) {
	current_test->cksum_input << "flash " << plug_flash.name();

	auto flash_drive = IR::program->get_flash_drive_or_null(plug_flash.name());
	if (!flash_drive) {
		throw Exception(std::string(plug_flash.ast_node->begin()) + ": Error: unknown flash drive: " + plug_flash.name());
	}
	visit_flash(flash_drive);

}

void VisitorSemantic::visit_plug_dvd(const IR::PlugDVD& plug_dvd, bool is_on) {
	current_test->cksum_input << "dvd";

	if (is_on) {
		auto dvd_path = plug_dvd.path();
		if (!fs::exists(dvd_path)) {
			throw Exception(std::string(plug_dvd.ast_node->begin()) + ": Error: specified dvd image path does not exist: " + dvd_path.generic_string());
		}
		current_test->cksum_input << " " << dvd_path.generic_string()
			<< " (file signature = " << file_signature(dvd_path) << ")";
	}
}

void VisitorSemantic::visit_plug_nic(const IR::PlugNIC& plug_nic) {
	current_test->cksum_input << "nic " << plug_nic.name();
}

void VisitorSemantic::visit_plug_link(const IR::PlugLink& plug_link) {
	current_test->cksum_input << "link " << plug_link.name();
}

void VisitorSemantic::visit_plug_hostdev(const IR::PlugHostDev& plug_hostdev) {
	current_test->cksum_input << "hostdev " << plug_hostdev.type() << " \"" << plug_hostdev.addr() << "\"";

	if (env->hypervisor() == "hyperv") {
		throw std::runtime_error(std::string(plug_hostdev.ast_node->begin()) + ": Sorry, Hyper-V does not support this command");
	}

	try {
		parse_usb_addr(plug_hostdev.addr());
	} catch (const std::exception& error) {
		throw Exception(std::string(plug_hostdev.ast_node->begin()) + ": Error: spicified usb addr is not valid: " + plug_hostdev.addr());
	}
}

void VisitorSemantic::visit_start(const IR::Start& start) {
	current_test->cksum_input << "start" << std::endl;
}

void VisitorSemantic::visit_stop(const IR::Stop& stop) {
	current_test->cksum_input << "stop" << std::endl;
}

void VisitorSemantic::visit_shutdown(const IR::Shutdown& shutdown) {
	current_test->cksum_input << "shutdown timeout " << shutdown.timeout() << std::endl;
}

void VisitorSemantic::visit_exec(const IR::Exec& exec) {
	if ((exec.interpreter() != "bash") &&
		(exec.interpreter() != "cmd") &&
		(exec.interpreter() != "python") &&
		(exec.interpreter() != "python2") &&
		(exec.interpreter() != "python3"))
	{
		throw Exception(std::string(exec.ast_node->begin()) + ": Error: unknown process name: " + exec.interpreter());
	}

	current_test->cksum_input << "exec "
		<< exec.interpreter() << " \"\"\"" << exec.script() << "\"\"\""
		<< " timeout " << exec.timeout()
		<< std::endl;
}

void VisitorSemantic::visit_copy(const IR::Copy& copy) {
	if (copy.ast_node->is_to_guest()) {
		current_test->cksum_input << "copyto ";
	} else {
		current_test->cksum_input << "copyfrom ";
	}

	current_test->cksum_input << copy.from()<< " " << copy.to() << " timeout " << copy.timeout() << std::endl;

	auto from = copy.from();
	if (copy.ast_node->is_to_guest()) {

		if (!copy.ast_node->nocheck) {
			if (!fs::exists(from)) {
				throw Exception(std::string(copy.ast_node->begin()) + ": Error: specified path doesn't exist: " + from);
			}

			current_test->cksum_input << pretty_files_signature(from) << std::endl;
		} else {
			current_test->cksum_input << " nocheck" << std::endl;
		}

	} else {
		if (copy.ast_node->nocheck) {
			throw Exception(std::string(copy.ast_node->nocheck.begin()) + ": Error: \"nocheck\" specifier is not applicable to copyfrom action");
		}
	}
}

void VisitorSemantic::visit_screenshot(const IR::Screenshot& screenshot) {
	auto destination = screenshot.destination();
	current_test->cksum_input << "screenshot" << " " << destination << std::endl;

	//No additional checks needed
}

void VisitorSemantic::visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr) {
	if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::ISelectable>>(select_expr)) {
		return visit_detect_selectable(p->select_expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectBinOp>>(select_expr)) {
		return visit_detect_binop(p->select_expr);
	} else {
		throw Exception("Unknown detect expr type");
	}
}

void VisitorSemantic::visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable) {
	bool is_negated = selectable->is_negated();
	if (is_negated) {
		current_test->cksum_input << "!";
	}

	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectText>>(selectable)) {
		visit_select_text({p->selectable, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(selectable)) {
		visit_select_js({p->selectable, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectImg>>(selectable)) {
		visit_select_img({p->selectable, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectHomm3>>(selectable)) {
		visit_select_homm3({p->selectable, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectParentedExpr>>(selectable)) {
		visit_detect_parented(p->selectable);
	} else {
		throw Exception("Unknown selectable type");
	}
}

void VisitorSemantic::visit_detect_parented(std::shared_ptr<AST::SelectParentedExpr> parented) {
	current_test->cksum_input << "(";
	visit_detect_expr(parented->select_expr);
	current_test->cksum_input << ")";

}

void VisitorSemantic::visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop) {
	visit_detect_expr(binop->left);
	current_test->cksum_input << binop->t.value();
	visit_detect_expr(binop->right);
}

void VisitorSemantic::visit_wait(const IR::Wait& wait) {
	current_test->cksum_input << "wait ";
	visit_detect_expr(wait.ast_node->select_expr);
	current_test->cksum_input << " timeout " << wait.timeout()
		<< " interval " << wait.interval()
		<< std::endl;
}

void VisitorSemantic::visit_sleep(const IR::Sleep& sleep) {
	current_test->cksum_input << "sleep timeout " << sleep.timeout() << std::endl;
}

void VisitorSemantic::visit_cmd_macro_call(const IR::MacroCall& macro_call) {
	macro_call.visit_semantic<AST::MacroBodyCommand>(this);
}

void VisitorSemantic::visit_action_macro_call(const IR::MacroCall& macro_call) {
	macro_call.visit_semantic<AST::MacroBodyAction>(this);
}

void VisitorSemantic::visit_macro_body(const std::shared_ptr<AST::MacroBodyCommand>& macro_body) {
	visit_command_block(macro_body->cmd_block);
}

void VisitorSemantic::visit_macro_body(const std::shared_ptr<AST::MacroBodyAction>& macro_body) {
	visit_action_block(macro_body->action_block->action);
}


Tribool VisitorSemantic::visit_expr(std::shared_ptr<AST::IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::BinOp>>(expr)) {
		return visit_binop(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::IFactor>>(expr)) {
		return visit_factor(p->expr);
	} else {
		throw Exception("Unknown expr type");
	}
}


Tribool VisitorSemantic::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	auto left = visit_expr(binop->left);
	current_test->cksum_input << " " << binop->op().value() << " ";

	if (binop->op().value() == "AND") {
		if (left == Tribool::no) {
			return left;
		} else {
			return visit_expr(binop->right);
		}
	} else if (binop->op().value() == "OR") {
		if (left == Tribool::yes) {
			return left;
		} else {
			return visit_expr(binop->right);
		}
	} else {
		throw Exception("Unknown binop operation");
	}
}

Tribool VisitorSemantic::visit_defined(const IR::Defined& defined) {
	bool is_defined = defined.is_defined();

	current_test->cksum_input
		<< "DEFINED " << defined.var();

	return is_defined ? Tribool::yes : Tribool::no;
}

Tribool VisitorSemantic::visit_comparison(const IR::Comparison& comparison) {
	current_test->cksum_input << comparison.left() << comparison.op() << comparison.right();

	return comparison.calculate() ? Tribool::yes : Tribool::no;
}

Tribool VisitorSemantic::visit_factor(std::shared_ptr<AST::IFactor> factor) {
	bool is_negated = factor->is_negated();
	if (is_negated) {
		current_test->cksum_input << "NOT ";
	}

	if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Check>>(factor)) {
		return visit_check({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::IExpr>>(factor)) {
		return is_negated ^ visit_expr(p->factor);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Defined>>(factor)) {
		return is_negated ^ visit_defined({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::Comparison>>(factor)) {
		return is_negated ^ visit_comparison({p->factor, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::ParentedExpr>>(factor)) {
		return is_negated ^ visit_parented_expr(p->factor);
	} else if (auto p = std::dynamic_pointer_cast<AST::Factor<AST::String>>(factor)) {
		try {
			auto text = template_parser.resolve(p->factor->text(), stack);
			current_test->cksum_input << "\"" << text << "\"";
			return is_negated ^ (text.length() ? Tribool::yes : Tribool::no);
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(p->factor->begin(), p->factor->text()));
		}
	} else {
		throw Exception("Unknown factor type");
	}
}

Tribool VisitorSemantic::visit_parented_expr(std::shared_ptr<AST::ParentedExpr> parented) {
	current_test->cksum_input << "(";
	auto result = visit_expr(parented->expr);
	current_test->cksum_input << ")";
	return result;
}

Tribool VisitorSemantic::visit_check(const IR::Check& check) {
	if (std::dynamic_pointer_cast<IR::FlashDrive>(current_controller)) {
		throw Exception(std::string(check.ast_node->begin()) + ": Error: The \"check\" expression is not applicable to a flash drive");
	}

	current_test->cksum_input << "check ";
	visit_detect_expr(check.ast_node->select_expr);
	current_test->cksum_input
		<< " timeout " << check.timeout()
		<< " interval " << check.interval();
	return Tribool::maybe;
}

void VisitorSemantic::visit_if_clause(std::shared_ptr<AST::IfClause> if_clause) {
	current_test->cksum_input << "if (";

	auto expr_value = visit_expr(if_clause->expr);

	current_test->cksum_input << ") {" << std::endl;

	switch (expr_value) {
		case Tribool::yes:
			visit_action(if_clause->if_action);
			break;
		case Tribool::no:
			if (if_clause->has_else()) {
				current_test->cksum_input << "} else {" << std::endl;
				visit_action(if_clause->else_action);
			}
			break;
		default:
			visit_action(if_clause->if_action);
			if (if_clause->has_else()) {
				current_test->cksum_input << "} else {" << std::endl;
				visit_action(if_clause->else_action);
			}
			break;
	}

	current_test->cksum_input << "}" << std::endl;
}

std::vector<std::string> VisitorSemantic::visit_range(const IR::Range& range) {
	std::string r1 = range.r1();
	std::string r2 = range.r2();

	if (!is_number(r1)) {
		throw Exception(std::string(range.ast_node->begin()) + ": Error: Can't convert range start " + r1 + " to a non-negative number");
	}

	auto r1_num = std::stoi(r1);

	if (r1_num < 0) {
		throw Exception(std::string(range.ast_node->begin()) + ": Error: Can't convert range start " + r1 + " to a non-negative number");
	}

	if (!is_number(r2)) {
		throw Exception(std::string(range.ast_node->begin()) + ": Error: Can't convert range finish " + r2 + " to a non-negative number");
	}

	auto r2_num = std::stoi(r2);

	if (r2_num < 0) {
		throw Exception(std::string(range.ast_node->begin()) + ": Error: Can't convert range finish " + r2 + " to a non-negative number");
	}

	if (r1_num >= r2_num) {
		throw Exception(std::string(range.ast_node->begin()) + ": Error: start of the range " +
			r1 + " is greater or equal to finish " + r2);
	}

	return range.values();
}

void VisitorSemantic::visit_for_clause(std::shared_ptr<AST::ForClause> for_clause) {
	current_test->cksum_input << "for (";
	std::vector<std::string> values;
	if (auto p = std::dynamic_pointer_cast<AST::CounterList<AST::Range>>(for_clause->counter_list)) {
		values = visit_range({p->counter_list, stack});
	} else {
		throw Exception("Unknown counter list type");
	}
	current_test->cksum_input << ") {" << std::endl;

	std::map<std::string, std::string> vars;
	for (auto i: values) {
		vars[for_clause->counter.value()] = i;
		auto new_stack = std::make_shared<StackNode>();
		new_stack->parent = stack;
		new_stack->vars = vars;
		StackPusher<VisitorSemantic> new_ctx(this, new_stack);
		visit_action(for_clause->cycle_body);
	}

	if (for_clause->else_token) {
		current_test->cksum_input << "} else {" << std::endl;
		visit_action(for_clause->else_action);
	}

	current_test->cksum_input << "}" << std::endl;
}

void VisitorSemantic::visit_cycle_control(const IR::CycleControl& cycle_control) {
	current_test->cksum_input << cycle_control.type() << std::endl;
}

void VisitorSemantic::visit_machine(std::shared_ptr<IR::Machine> machine) {
	try {
		current_test->mentioned_machines.insert(machine);

		auto result = visited_machines.insert(machine);
		if (!result.second) {
			return;
		}

		StackPusher<VisitorSemantic> new_ctx(this, machine->stack);

		machine->config = visit_attr_block(machine->ast_node->attr_block, "vm_global");
		machine->config["prefix"] = prefix;
		machine->config["name"] = machine->name();
		machine->config["src_file"] = machine->ast_node->name->begin().file.generic_string();

		machine->validate_config();
	} catch (const std::exception& error) {
		std::throw_with_nested(ControllerCreatonException(machine));
	}
}

void VisitorSemantic::visit_flash(std::shared_ptr<IR::FlashDrive> flash) {
	try {
		current_test->mentioned_flash_drives.insert(flash);

		auto result = visited_flash_drives.insert(flash);
		if (!result.second) {
			return;
		}

		StackPusher<VisitorSemantic> new_ctx(this, flash->stack);

		//no need to check for duplicates
		//It's already done in Parser while registering Controller
		flash->config = visit_attr_block(flash->ast_node->attr_block, "fd_global");
		flash->config["prefix"] = prefix;
		flash->config["name"] = flash->name();
		flash->config["src_file"] = flash->ast_node->name->begin().file.generic_string();

		flash->validate_config();
	} catch (const std::exception& error) {
		std::throw_with_nested(ControllerCreatonException(flash));
	}
}

void VisitorSemantic::visit_network(std::shared_ptr<IR::Network> network) {
	try {
		current_test->mentioned_networks.insert(network);

		auto result = visited_networks.insert(network);
		if (!result.second) {
			return;
		}

		StackPusher<VisitorSemantic> new_ctx(this, network->stack);

		network->config = visit_attr_block(network->ast_node->attr_block, "network_global");
		network->config["prefix"] = prefix;
		network->config["name"] = network->name();
		network->config["src_file"] = network->ast_node->name->begin().file.generic_string();

		network->validate_config();
	} catch (const std::exception& error) {
		std::throw_with_nested(ControllerCreatonException(network));
	}
}

nlohmann::json VisitorSemantic::visit_attr_block(std::shared_ptr<AST::AttrBlock> attr_block, const std::string& ctx_name) {
	nlohmann::json config;
	for (auto attr: attr_block->attrs) {
		if (config.count(attr->name.value())) {
			if (!config.at(attr->name.value()).is_array()) {
				throw Exception(std::string(attr->begin()) + ": Error: duplicate attribute: \"" + attr->name.value() + "\"");
			}
		}
		nlohmann::json j = visit_attr(attr, ctx_name);
		if (attr->id) {
			j["name"] = attr->id.value();
			config[attr->name.value()].push_back(j);
		}  else {
			config[attr->name.value()] = j;
		}
	}
	return config;
}

nlohmann::json VisitorSemantic::visit_attr(std::shared_ptr<AST::Attr> attr, const std::string& ctx_name) {
	auto ctx = attr_ctxs.find(ctx_name);
	if (ctx == attr_ctxs.end()) {
		throw Exception("Unknown ctx"); //should never happen
	}

	auto found = ctx->second.find(attr->name);

	if (found == ctx->second.end()) {
		throw Exception(std::string(attr->begin()) + ": Error: unknown attribute name: \"" + attr->name.value() + "\"");
	}

	auto attr_meta = found->second;
	if (attr->id != attr_meta.name_is_required) {
		if (attr_meta.name_is_required) {
			throw Exception(std::string(attr->end()) + ": Error: attribute \"" + attr->name.value() +
				"\" requires a name");
		} else {
			throw Exception(std::string(attr->end()) + ": Error: attribute \"" + attr->name.value() +
				"\" must have no name");
		}
	}

	if ((attr->value->type() != attr_meta.type)) {
		if (attr_meta.type == Token::category::attr_block) {
			throw Exception(std::string(attr->end()) + ": Error: unexpected value type \"" +
				Token::type_to_string(attr->value->type()) + "\" for attribute \"" + attr->name.value() +
				"\", expected \"" + Token::type_to_string(attr_meta.type) + "\"");
		}

		if (attr->value->type() != Token::category::quoted_string) {
			throw Exception(std::string(attr->end()) + ": Error: unexpected value type \"" +
				Token::type_to_string(attr->value->type()) + "\" for attribute \"" + attr->name.value() +
				"\", expected \"" + Token::type_to_string(attr_meta.type) + "\" OR \"" +
				Token::type_to_string(Token::category::quoted_string) + "\"");
		}
	}

	if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::SimpleAttr>>(attr->value)) {
		//strings are here too, we treat them separately

		//1) If we expect string -> check that it's string
		if (attr_meta.type == Token::category::quoted_string) {
			try {
				return template_parser.resolve(p->attr_value->value->text(), stack);
			} catch (const std::exception& error) {
				std::throw_with_nested(ResolveException(p->attr_value->value->begin(), p->attr_value->value->text()));
			}
		} else {
			p->attr_value->value->expected_token_type = attr_meta.type;
			std::string value;
			try {
				value = IR::StringTokenUnion(p->attr_value->value, stack).resolve();
			} catch (const std::exception& error) {
				throw Exception(std::string(attr->end()) + ": Error: can't convert value string \"" +
					p->attr_value->value->text() + "\" for attribute \"" + attr->name.value() + "\" into expected \"" +
					Token::type_to_string(attr_meta.type) + "\"");
			}

			if (p->attr_value->type() == Token::category::number) {
				if (std::stoi(value) < 0) {
					throw Exception(std::string(attr->begin()) + ": Error: numeric attr can't be negative: " + value);
				}
				return std::stoul(value);
			} else if (p->attr_value->type() == Token::category::size) {
				return size_to_mb(value);
			} else if (p->attr_value->type() == Token::category::boolean) {
				return str_to_bool(value);
			} else {
				throw Exception(std::string(attr->begin()) + ": Error: unsupported attr: " + value);
			}
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::AttrBlock>>(attr->value)) {
		return visit_attr_block(p->attr_value, attr->name);
	} else {
		throw Exception("Unknown attr category");
	}
}

bool VisitorSemantic::is_button(const Token& t) const {
	std::string button = t.value();
	std::transform(button.begin(), button.end(), button.begin(), ::toupper);
	return (keys.find(button) != keys.end());
}
