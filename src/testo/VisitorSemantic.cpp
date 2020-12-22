
#include "VisitorSemantic.hpp"
#include <nn/Homm3Object.hpp>
#include "backends/Environment.hpp"
#include "Exceptions.hpp"
#include "IR/Program.hpp"
#include "Parser.hpp"
#include "js/Context.hpp"
#include <fmt/format.h>
#include <wildcards.hpp>

void VisitorSemanticConfig::validate() const {

}

VisitorSemantic::VisitorSemantic(const VisitorSemanticConfig& config) {
	prefix = config.prefix;

	keys.insert("ESC");
	keys.insert("ONE");
	keys.insert("TWO");
	keys.insert("THREE");
	keys.insert("FOUR");
	keys.insert("FIVE");
	keys.insert("SIX");
	keys.insert("SEVEN");
	keys.insert("EIGHT");
	keys.insert("NINE");
	keys.insert("ZERO");
	keys.insert("MINUS");
	keys.insert("EQUALSIGN");
	keys.insert("BACKSPACE");
	keys.insert("TAB");
	keys.insert("Q");
	keys.insert("W");
	keys.insert("E");
	keys.insert("R");
	keys.insert("T");
	keys.insert("Y");
	keys.insert("U");
	keys.insert("I");
	keys.insert("O");
	keys.insert("P");
	keys.insert("LEFTBRACE");
	keys.insert("RIGHTBRACE");
	keys.insert("ENTER");
	keys.insert("LEFTCTRL");
	keys.insert("A");
	keys.insert("S");
	keys.insert("D");
	keys.insert("F");
	keys.insert("G");
	keys.insert("H");
	keys.insert("J");
	keys.insert("K");
	keys.insert("L");
	keys.insert("SEMICOLON");
	keys.insert("APOSTROPHE");
	keys.insert("GRAVE");
	keys.insert("LEFTSHIFT");
	keys.insert("BACKSLASH");
	keys.insert("Z");
	keys.insert("X");
	keys.insert("C");
	keys.insert("V");
	keys.insert("B");
	keys.insert("N");
	keys.insert("M");
	keys.insert("COMMA");
	keys.insert("DOT");
	keys.insert("SLASH");
	keys.insert("RIGHTSHIFT");
	keys.insert("LEFTALT");
	keys.insert("SPACE");
	keys.insert("CAPSLOCK");
	keys.insert("F1"),
	keys.insert("F2"),
	keys.insert("F3"),
	keys.insert("F4"),
	keys.insert("F5"),
	keys.insert("F6"),
	keys.insert("F7"),
	keys.insert("F8"),
	keys.insert("F9"),
	keys.insert("F10"),
	keys.insert("F11"),
	keys.insert("F12"),
	keys.insert("NUMLOCK");
	keys.insert("KP_0");
	keys.insert("KP_1");
	keys.insert("KP_2");
	keys.insert("KP_3");
	keys.insert("KP_4");
	keys.insert("KP_5");
	keys.insert("KP_6");
	keys.insert("KP_7");
	keys.insert("KP_8");
	keys.insert("KP_9");
	keys.insert("KP_PLUS");
	keys.insert("KP_MINUS");
	keys.insert("KP_SLASH");
	keys.insert("KP_ASTERISK");
	keys.insert("KP_ENTER");
	keys.insert("KP_DOT");
	keys.insert("SCROLLLOCK");
	keys.insert("RIGHTCTRL");
	keys.insert("RIGHTALT");
	keys.insert("HOME");
	keys.insert("UP");
	keys.insert("PAGEUP");
	keys.insert("LEFT");
	keys.insert("RIGHT");
	keys.insert("END");
	keys.insert("DOWN");
	keys.insert("PAGEDOWN");
	keys.insert("INSERT");
	keys.insert("DELETE");
	keys.insert("SCROLLUP");
	keys.insert("SCROLLDOWN");
	keys.insert("LEFTMETA");
	keys.insert("RIGHTMETA");

	//init attr ctx
	attr_ctx vm_global_ctx;
	vm_global_ctx.insert({"ram", std::make_pair(false, Token::category::size)});
	vm_global_ctx.insert({"iso", std::make_pair(false, Token::category::quoted_string)});
	vm_global_ctx.insert({"nic", std::make_pair(true, Token::category::attr_block)});
	vm_global_ctx.insert({"disk", std::make_pair(true, Token::category::attr_block)});
	vm_global_ctx.insert({"video", std::make_pair(true, Token::category::attr_block)});
	vm_global_ctx.insert({"cpus", std::make_pair(false, Token::category::number)});
	vm_global_ctx.insert({"qemu_spice_agent", std::make_pair(false, Token::category::binary)});
	vm_global_ctx.insert({"loader", std::make_pair(false, Token::category::quoted_string)});

	attr_ctxs.insert({"vm_global", vm_global_ctx});

	attr_ctx disk_ctx;
	disk_ctx.insert({"size", std::make_pair(false, Token::category::size)});
	disk_ctx.insert({"source", std::make_pair(false, Token::category::quoted_string)});

	attr_ctxs.insert({"disk", disk_ctx});

	attr_ctx vm_network_ctx;
	vm_network_ctx.insert({"slot", std::make_pair(false, Token::category::number)});
	vm_network_ctx.insert({"attached_to", std::make_pair(false, Token::category::quoted_string)});
	vm_network_ctx.insert({"mac", std::make_pair(false, Token::category::quoted_string)});
	vm_network_ctx.insert({"adapter_type", std::make_pair(false, Token::category::quoted_string)});

	attr_ctxs.insert({"nic", vm_network_ctx});

	attr_ctx video_ctx;
	video_ctx.insert({"qemu_mode", std::make_pair(false, Token::category::quoted_string)});

	attr_ctxs.insert({"video", video_ctx});

	attr_ctx fd_global_ctx;
	fd_global_ctx.insert({"fs", std::make_pair(false, Token::category::quoted_string)});
	fd_global_ctx.insert({"size", std::make_pair(false, Token::category::size)});
	fd_global_ctx.insert({"folder", std::make_pair(false, Token::category::quoted_string)});

	attr_ctxs.insert({"fd_global", fd_global_ctx});

	attr_ctx network_global_ctx;
	network_global_ctx.insert({"mode", std::make_pair(false, Token::category::quoted_string)});
	network_global_ctx.insert({"persistent", std::make_pair(false, Token::category::binary)});
	network_global_ctx.insert({"autostart", std::make_pair(false, Token::category::binary)});

	attr_ctxs.insert({"network_global", network_global_ctx});

	attr_ctx test_global_ctx;
	test_global_ctx.insert({"no_snapshots", std::make_pair(false, Token::category::binary)});
	test_global_ctx.insert({"description", std::make_pair(false, Token::category::quoted_string)});
	attr_ctxs.insert({"test_global", test_global_ctx});
}

static uint32_t size_to_mb(const std::string& size) {
	uint32_t result = std::stoul(size.substr(0, size.length() - 2));
	if (size[size.length() - 2] == 'M') {
		result = result * 1;
	} else if (size[size.length() - 2] == 'G') {
		result = result * 1024;
	} else {
		throw std::runtime_error("Unknown size specifier"); //should not happen ever
	}

	return result;
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
					throw std::runtime_error(std::string(test->ast_node->begin()) + ": Error: some parents have common virtual machines");
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
					throw std::runtime_error(std::string(test->ast_node->begin()) + ": Error: some parents have common flash drives");
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

	StackPusher<VisitorSemantic> new_ctx(this, macro->stack);

	for (size_t i = 0; i < macro->ast_node->args.size(); ++i) {
		for (size_t j = i + 1; j < macro->ast_node->args.size(); ++j) {
			if (macro->ast_node->args[i]->name() == macro->ast_node->args[j]->name()) {
				throw std::runtime_error(std::string(macro->ast_node->args[j]->begin()) + ": Error: duplicate macro arg: " + macro->ast_node->args[j]->name());
			}
		}
	}

	bool has_default = false;
	for (auto arg: macro->ast_node->args) {
		if (arg->default_value) {
			has_default = true;
			continue;
		}

		if (has_default && !arg->default_value) {
			throw std::runtime_error(std::string(arg->begin()) + ": Error: default value must be specified for macro arg " + arg->name());
		}
	}
}

void VisitorSemantic::visit_test(std::shared_ptr<IR::Test> test) {
	if (test->ast_node->attrs) {
		test->attrs = visit_attr_block(test->ast_node->attrs, "test_global");
	}

	current_test = test;

	current_test->cksum_input = test->name();
	std::vector<std::string> parents_names;
	for (auto parent: test->parents) {
		parents_names.push_back(parent->name());
	}
	std::sort(parents_names.begin(), parents_names.end());
	for (auto name: parents_names) {
		current_test->cksum_input += name;
	}
	current_test->cksum_input += test->snapshots_needed();

	StackPusher<VisitorSemantic> new_ctx(this, test->stack);
	visit_command_block(test->ast_node->cmd_block);

	std::hash<std::string> h;
	current_test->cksum = std::to_string(h(current_test->cksum_input));

	current_test = nullptr;
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
		visit_macro_call(p->cmd, true);
	} else {
		throw std::runtime_error("Should never happen");
	}
}

void VisitorSemantic::visit_regular_command(const IR::RegularCommand& regular_cmd) {
	current_test->cksum_input += regular_cmd.entity();
	if ((current_controller = IR::program->get_machine_or_null(regular_cmd.entity()))) {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		visit_machine(vmc);

		if (vmc->config.count("nic")) {
			auto nics = vmc->config.at("nic");
			for (auto& nic: nics) {
				if (nic.count("attached_to")) {
					std::string network_name = nic.at("attached_to");
					auto network = IR::program->get_network_or_null(network_name);
					if (!network) {
						throw std::runtime_error(fmt::format("Can't construct VmController for vm \"{}\": nic \"{}\" is attached to an unknown network: \"{}\"",
							vmc->config.at("name").get<std::string>(), nic.at("name").get<std::string>(), network_name));
					}
					visit_network(network);
				}
			}
		}
		visit_action_vm(regular_cmd.ast_node->action);
	} else if ((current_controller = IR::program->get_flash_drive_or_null(regular_cmd.entity()))) {
		auto fdc = std::dynamic_pointer_cast<IR::FlashDrive>(current_controller);
		visit_flash(fdc);
		visit_action_fd(regular_cmd.ast_node->action);
	} else {
		throw std::runtime_error(std::string(regular_cmd.ast_node->entity->begin()) + ": Error: unknown virtual entity: " + regular_cmd.entity());
	}

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
		return visit_abort({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		return visit_print({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Type>>(action)) {
		return visit_type({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Press>>(action)) {
		return visit_press({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Hold>>(action)) {
		return visit_hold({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Release>>(action)) {
		return visit_release({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		return visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Mouse>>(action)) {
		return visit_mouse({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Plug>>(action)) {
		return visit_plug({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Start>>(action)) {
		return visit_start({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Stop>>(action)) {
		return visit_stop({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Shutdown>>(action)) {
		return visit_shutdown({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Exec>>(action)) {
		return visit_exec({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		return visit_copy({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Wait>>(action)) {
		return visit_wait({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		return visit_sleep({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroCall>>(action)) {
		return visit_macro_call(p->action, false);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		return visit_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		return visit_for_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		return visit_cycle_control({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
		;
	} else {
		throw std::runtime_error(std::string(action->begin()) + ": Error: The action \"" + action->t.value() + "\" is not applicable to a virtual machine");
	}
}

void VisitorSemantic::visit_action_fd(std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Abort>>(action)) {
		return visit_abort({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		return visit_print({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Copy>>(action)) {
		return visit_copy({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		return visit_sleep({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		return visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
		;
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroCall>>(action)) {
		return visit_macro_call(p->action, false);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		return visit_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		return visit_for_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		return visit_cycle_control({p->action, stack});
	} else {
		throw std::runtime_error(std::string(action->begin()) + ": Error: The action \"" + action->t.value() + "\" is not applicable to a flash drive");
	}
}

void VisitorSemantic::visit_abort(const IR::Abort& abort) {
	current_test->cksum_input += "abort ";
	current_test->cksum_input += abort.message();
}

void VisitorSemantic::visit_print(const IR::Print& print) {
	current_test->cksum_input += "print ";
	current_test->cksum_input += print.message();
}

void VisitorSemantic::visit_type(const IR::Type& type) {
	current_test->cksum_input += "type ";
	current_test->cksum_input += type.text();
	current_test->cksum_input += type.interval();
}

void VisitorSemantic::visit_press(const IR::Press& press) {
	for (auto key_spec: press.ast_node->keys) {
		visit_key_spec(key_spec);
	}

	current_test->cksum_input += std::string(*press.ast_node);
	current_test->cksum_input += press.interval();
}

void VisitorSemantic::visit_key_combination(std::shared_ptr<AST::KeyCombination> combination) {
	for (size_t i = 0; i < combination->buttons.size(); ++i) {
		auto button = combination->buttons[i];
		if (!is_button(button)) {
			throw std::runtime_error(std::string(button.begin()) +
				" :Error: unknown key: " + button.value());
		}

		for (size_t j = i + 1; j < combination->buttons.size(); ++j) {
			if (button.value() == combination->buttons[j].value()) {
				throw std::runtime_error(std::string(combination->buttons[j].begin()) +
					" :Error: duplicate key: " + button.value());
			}
		}
	}
}

void VisitorSemantic::visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec) {
	visit_key_combination(key_spec->combination);

	if (key_spec->times.value().length()) {
		if (std::stoi(key_spec->times.value()) < 1) {
			throw std::runtime_error(std::string(key_spec->times.begin()) +
					" :Error: can't press a button less than 1 time: " + key_spec->times.value());
		}
	}
}

void VisitorSemantic::visit_hold(const IR::Hold& hold) {
	current_test->cksum_input += std::string(*hold.ast_node);
	visit_key_combination(hold.ast_node->combination);
}

void VisitorSemantic::visit_release(const IR::Release& release) {
	current_test->cksum_input += std::string(*release.ast_node);
	if (release.ast_node->combination) {
		visit_key_combination(release.ast_node->combination);
	}
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

		current_test->cksum_input += std::string(*specifier);
		if (specifier->is_from()) {
			if (!arg) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}

			auto i = std::stoi(arg.value());
			if (i < 0) {
				throw std::runtime_error(std::string(arg.begin()) + ": Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}

			if (has_from) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after another \"from\" specifier");
			}
			if (has_center) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after a \"precision\" specifier");
			}
			if (has_move) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after a \"move\" specifier");
			}
			has_from = true;
			continue;
		} if (specifier->is_centering()) {
			if (arg) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: specifier " + specifier->name.value() + " must not have an argument");
			}
			if (has_center) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after another \"precision\" specifier");
			}
			if (has_move) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: you can't use specifier " + specifier->name.value() + " after a \"move\" specifier");
			}
			has_center = true;
			continue;
		} else if (specifier->is_moving()) {
			if (!arg) {
				throw std::runtime_error(std::string(specifier->begin()) + ": Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}

			auto i = std::stoi(arg.value());
			if (i < 0) {
				throw std::runtime_error(std::string(arg.begin()) + ": Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}
			has_move = true;
			continue;
		} else {
			throw std::runtime_error(std::string(specifier->begin()) + ": Error: unknown specifier: " + specifier->name.value());
		}

	}
}

void VisitorSemantic::visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates) {
	current_test->cksum_input += "coordinates x:";
	current_test->cksum_input += coordinates.x();
	current_test->cksum_input += " y:";
	current_test->cksum_input += coordinates.y();
}

void VisitorSemantic::visit_select_js(const IR::SelectJS& js) {
	auto script = js.script();

	if (!script.length()) {
		throw std::runtime_error(std::string(js.ast_node->begin()) + ": Error: empty script in js selection");
	}

	try {
		validate_js(script);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(std::string(js.ast_node->begin()) + ": Error while validating js selection"));
	}

	current_test->cksum_input += "js ";
	current_test->cksum_input += script;
}

void VisitorSemantic::visit_select_img(const IR::SelectImg& img) {
	auto img_path = img.img_path();

	if (!fs::exists(img_path)) {
		throw std::runtime_error(std::string(img.ast_node->begin()) + ": Error: specified image path does not exist: " + img_path.generic_string());
	}

	if (!fs::is_regular_file(img_path)) {
		throw std::runtime_error(std::string(img.ast_node->begin()) + ": Error: specified image path does not lead to a regular file: " + img_path.generic_string());
	}

	current_test->cksum_input += "img ";
	current_test->cksum_input += img_path.generic_string();
	current_test->cksum_input += file_signature(img_path);
}

void VisitorSemantic::visit_select_homm3(const IR::SelectHomm3& homm3) {
	auto id = homm3.id();

	if (!nn::Homm3Object::check_class_name(id)) {
		throw std::runtime_error(std::string(homm3.ast_node->begin()) + ": Error: specified Heroes of Might and Magic object does not exist " + id);
	}

	current_test->cksum_input += "homm3 ";
	current_test->cksum_input += id;
}

void VisitorSemantic::visit_select_text(const IR::SelectText& text) {
	auto txt = text.text();
	if (!txt.length()) {
		throw std::runtime_error(std::string(text.ast_node->begin()) + ": Error: empty string in text selection");
	}

	current_test->cksum_input += "text ";
	current_test->cksum_input += txt;
}

void VisitorSemantic::visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable) {
	if (mouse_selectable.ast_node->selectable->is_negated()) {
		throw std::runtime_error(std::string(mouse_selectable.ast_node->begin()) + ": Error: negation is not supported for mouse move/click actions");
	}

	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(mouse_selectable.ast_node->selectable)) {
		if (mouse_selectable.ast_node->specifiers.size()) {
			throw std::runtime_error(std::string(mouse_selectable.ast_node->specifiers[0]->begin()) + ": Error: mouse specifiers are not supported for js selections");
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
		throw std::runtime_error(std::string(mouse_selectable.ast_node->begin()) + ": Error: select expressions are not supported for mouse move/click actions");
	}
}

void VisitorSemantic::visit_mouse_move_click(const IR::MouseMoveClick& mouse_move_click) {
	current_test->cksum_input += mouse_move_click.event_type() + " ";
	if (mouse_move_click.ast_node->object) {
		if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseCoordinates>>(mouse_move_click.ast_node->object)) {
			visit_mouse_move_coordinates({p->target, stack});
		} else if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseSelectable>>(mouse_move_click.ast_node->object)) {
			visit_mouse_move_selectable({p->target, stack});
		}
	}
}

void VisitorSemantic::visit_mouse_hold(const IR::MouseHold& mouse_hold) {
	current_test->cksum_input += "hold ";
	current_test->cksum_input += mouse_hold.button();
}

void VisitorSemantic::visit_mouse_release(const IR::MouseRelease& mouse_release) {
	current_test->cksum_input += "release";
}

void VisitorSemantic::visit_mouse(const IR::Mouse& mouse) {
	current_test->cksum_input += "mouse ";

	if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseMoveClick>>(mouse.ast_node->event)) {
		return visit_mouse_move_click({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseHold>>(mouse.ast_node->event)) {
		return visit_mouse_hold({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseRelease>>(mouse.ast_node->event)) {
		return visit_mouse_release({p->event, stack});
	}
}

void VisitorSemantic::visit_plug(const IR::Plug& plug) {
	current_test->cksum_input += "plug ";
	current_test->cksum_input += std::to_string(plug.is_on());
	current_test->cksum_input += plug.entity_type();

	if (plug.entity_type() == "dvd" && plug.is_on()) {
		auto dvd_path = plug.dvd_path();
		if (!fs::exists(dvd_path)) {
			throw std::runtime_error(std::string(plug.ast_node->begin()) + ": Error: specified dvd image path does not exist: " + dvd_path.generic_string());
		}
		current_test->cksum_input += dvd_path.generic_string();
		current_test->cksum_input += file_signature(dvd_path);
		return;
	}

	if (plug.entity_type() != "dvd") {
		current_test->cksum_input += plug.entity_name();
	}

	if (plug.entity_type() == "flash") {
		auto flash_drive = IR::program->get_flash_drive_or_null(plug.entity_name());
		if (!flash_drive) {
			throw std::runtime_error(std::string(plug.ast_node->begin()) + ": Error: unknown flash drive: " + plug.entity_name());
		}
		visit_flash(flash_drive);
	}
}

void VisitorSemantic::visit_start(const IR::Start& start) {
	current_test->cksum_input += "start";
}

void VisitorSemantic::visit_stop(const IR::Stop& stop) {
	current_test->cksum_input += "stop";
}

void VisitorSemantic::visit_shutdown(const IR::Shutdown& shutdown) {
	current_test->cksum_input += "shutdown ";
	current_test->cksum_input += shutdown.timeout();
}

void VisitorSemantic::visit_exec(const IR::Exec& exec) {
	if ((exec.interpreter() != "bash") &&
		(exec.interpreter() != "cmd") &&
		(exec.interpreter() != "python") &&
		(exec.interpreter() != "python2") &&
		(exec.interpreter() != "python3"))
	{
		throw std::runtime_error(std::string(exec.ast_node->begin()) + ": Error: unknown process name: " + exec.interpreter());
	}

	current_test->cksum_input += "exec ";
	current_test->cksum_input += exec.interpreter();
	current_test->cksum_input += exec.script();
	current_test->cksum_input += exec.timeout();
}

void VisitorSemantic::visit_copy(const IR::Copy& copy) {
	current_test->cksum_input += "copy ";
	current_test->cksum_input += copy.ast_node->is_to_guest();

	auto from = copy.from();

	if (copy.ast_node->is_to_guest()) {
		if (!fs::exists(from)) {
			throw std::runtime_error(std::string(copy.ast_node->begin()) + ": Error: specified path doesn't exist: " + from);
		}

		if (fs::is_regular_file(from)) {
			current_test->cksum_input += file_signature(from);
		} else if (fs::is_directory(from)) {
			current_test->cksum_input += directory_signature(from);
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(from).generic_string());
		}

		current_test->cksum_input += from;
	}

	current_test->cksum_input += copy.to();
	current_test->cksum_input += copy.timeout();
}

void VisitorSemantic::visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr) {
	if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::ISelectable>>(select_expr)) {
		return visit_detect_selectable(p->select_expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectBinOp>>(select_expr)) {
		return visit_detect_binop(p->select_expr);
	} else {
		throw std::runtime_error("Unknown detect expr type");
	}
}

void VisitorSemantic::validate_js(const std::string& script) {
	js::Context js_ctx(nullptr);
	js_ctx.eval(script, true);
}

void VisitorSemantic::visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable) {
	bool is_negated = selectable->is_negated();
	current_test->cksum_input += std::to_string(is_negated);

	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectText>>(selectable)) {
		visit_select_text({p->selectable, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(selectable)) {
		visit_select_js({p->selectable, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectImg>>(selectable)) {
		visit_select_img({p->selectable, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectParentedExpr>>(selectable)) {
		visit_detect_parented(p->selectable);
	} else {
		throw std::runtime_error("Unknown selectable type");
	}
}

void VisitorSemantic::visit_detect_parented(std::shared_ptr<AST::SelectParentedExpr> parented) {
	current_test->cksum_input += "(";
	visit_detect_expr(parented->select_expr);
	current_test->cksum_input += ")";

}

void VisitorSemantic::visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop) {
	visit_detect_expr(binop->left);
	current_test->cksum_input += binop->t.value();
	visit_detect_expr(binop->right);
}

void VisitorSemantic::visit_wait(const IR::Wait& wait) {
	current_test->cksum_input += "wait ";
	visit_detect_expr(wait.ast_node->select_expr);
	current_test->cksum_input += wait.timeout();
	current_test->cksum_input += wait.interval();
}

void VisitorSemantic::visit_sleep(const IR::Sleep& sleep) {
	current_test->cksum_input += "sleep ";
	current_test->cksum_input += sleep.timeout();
}

void VisitorSemantic::visit_macro_call(std::shared_ptr<AST::MacroCall> macro_call, bool is_command_macro) {
	auto macro = IR::program->get_macro_or_null(macro_call->name().value());
	if (!macro) {
		throw std::runtime_error(std::string(macro_call->begin()) + ": Error: unknown macro: " + macro_call->name().value());
	}

	visit_macro(macro);

	uint32_t args_with_default = 0;

	for (auto arg: macro->ast_node->args) {
		if (arg->default_value) {
			args_with_default++;
		}
	}

	if (macro_call->args.size() < macro->ast_node->args.size() - args_with_default) {
		throw std::runtime_error(fmt::format("{}: Error: expected at least {} args, {} provided", std::string(macro_call->begin()),
			macro->ast_node->args.size() - args_with_default, macro_call->args.size()));
	}

	if (macro_call->args.size() > macro->ast_node->args.size()) {
		throw std::runtime_error(fmt::format("{}: Error: expected at most {} args, {} provided", std::string(macro_call->begin()),
			macro->ast_node->args.size(), macro_call->args.size()));
	}

	std::map<std::string, std::string> vars;

	for (size_t i = 0; i < macro_call->args.size(); ++i) {
		try {
			auto value = template_parser.resolve(macro_call->args[i]->text(), stack);
			vars[macro->ast_node->args[i]->name()] = value;
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(macro_call->args[i]->begin(), macro_call->args[i]->text()));
		}
	}

	for (size_t i = macro_call->args.size(); i < macro->ast_node->args.size(); ++i) {
		try {
			auto value = template_parser.resolve(macro->ast_node->args[i]->default_value->text(), stack);
			vars[macro->ast_node->args[i]->name()] = value;
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(macro->ast_node->args[i]->default_value->begin(), macro->ast_node->args[i]->default_value->text()));
		}
	}

	StackPusher<VisitorSemantic> new_ctx(this, macro->new_stack(vars));

	if (!macro->ast_node->body) {
		Parser parser(macro->ast_node->body_tokens);
		if (is_command_macro) {
			auto cmd_block = parser.command_block();
			auto body = std::shared_ptr<AST::MacroBodyCommand>(new AST::MacroBodyCommand(cmd_block));
			macro->ast_node->body = std::shared_ptr<AST::MacroBody<AST::MacroBodyCommand>>(new AST::MacroBody<AST::MacroBodyCommand>(body));
		} else {
			//So it' supposed to be an action macro.
			auto action_block = parser.action_block();
			auto body = std::shared_ptr<AST::MacroBodyAction>(new AST::MacroBodyAction(action_block));
			macro->ast_node->body = std::shared_ptr<AST::MacroBody<AST::MacroBodyAction>>(new AST::MacroBody<AST::MacroBodyAction>(body));
		}
	}

	try {
		if (is_command_macro) {
			auto p = std::dynamic_pointer_cast<AST::MacroBody<AST::MacroBodyCommand>>(macro->ast_node->body);
			if (p == nullptr) {
				throw std::runtime_error(std::string(macro_call->begin()) + ": Error: the \"" + macro_call->name().value() + "\" macro does not contain commands, as expected");
			}

			visit_command_block(p->macro_body->cmd_block);
		} else {
			auto p = std::dynamic_pointer_cast<AST::MacroBody<AST::MacroBodyAction>>(macro->ast_node->body);
			if (p == nullptr) {
				throw std::runtime_error(std::string(macro_call->begin()) + ": Error: the \"" + macro_call->name().value() + "\" macro does not contain actions, as expected");
			}

			visit_action_block(p->macro_body->action_block->action);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(MacroException(macro_call));
	}
}

Tribool VisitorSemantic::visit_expr(std::shared_ptr<AST::IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::BinOp>>(expr)) {
		return visit_binop(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::IFactor>>(expr)) {
		return visit_factor(p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}


Tribool VisitorSemantic::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	auto left = visit_expr(binop->left);
	current_test->cksum_input += binop->op().value();

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
		throw std::runtime_error("Unknown binop operation");
	}
}

Tribool VisitorSemantic::visit_defined(const IR::Defined& defined) {
	current_test->cksum_input += "DEFINED ";
	current_test->cksum_input += defined.var();
	bool is_defined = defined.is_defined();
	current_test->cksum_input += is_defined;

	return is_defined ? Tribool::yes : Tribool::no;
}

Tribool VisitorSemantic::visit_comparison(const IR::Comparison& comparison) {
	current_test->cksum_input += comparison.left();
	current_test->cksum_input += comparison.op();
	current_test->cksum_input += comparison.right();

	return comparison.calculate() ? Tribool::yes : Tribool::no;
}

Tribool VisitorSemantic::visit_factor(std::shared_ptr<AST::IFactor> factor) {
	bool is_negated = factor->is_negated();
	current_test->cksum_input += std::to_string(is_negated);

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
			current_test->cksum_input += text;
			return is_negated ^ (text.length() ? Tribool::yes : Tribool::no);
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(p->factor->begin(), p->factor->text()));
		}
	} else {
		throw std::runtime_error("Unknown factor type");
	}
}

Tribool VisitorSemantic::visit_parented_expr(std::shared_ptr<AST::ParentedExpr> parented) {
	current_test->cksum_input += "(";
	auto result = visit_expr(parented->expr);
	current_test->cksum_input += ")";
	return result;
}

Tribool VisitorSemantic::visit_check(const IR::Check& check) {
	if (std::dynamic_pointer_cast<IR::FlashDrive>(current_controller)) {
		throw std::runtime_error(std::string(check.ast_node->begin()) + ": Error: The \"check\" expression is not applicable to a flash drive");
	}

	current_test->cksum_input += "check ";
	visit_detect_expr(check.ast_node->select_expr);
	current_test->cksum_input += check.timeout();
	current_test->cksum_input += check.interval();
	return Tribool::maybe;
}

void VisitorSemantic::visit_if_clause(std::shared_ptr<AST::IfClause> if_clause) {
	current_test->cksum_input += "if";

	auto expr_value = visit_expr(if_clause->expr);

	switch (expr_value) {
		case Tribool::yes:
			visit_action(if_clause->if_action);
			break;
		case Tribool::no:
			if (if_clause->has_else()) {
				visit_action(if_clause->else_action);
			}
			break;
		default:
			visit_action(if_clause->if_action);
			if (if_clause->has_else()) {
				current_test->cksum_input += "else";
				visit_action(if_clause->else_action);
			}
			break;
	}
}

std::vector<std::string> VisitorSemantic::visit_range(const IR::Range& range) {
	std::string r1 = range.r1();
	std::string r2 = range.r2();

	if (!is_number(r1)) {
		throw std::runtime_error(std::string(range.ast_node->begin()) + ": Error: Can't convert range start " + r1 + " to a non-negative number");
	}

	auto r1_num = std::stoi(r1);

	if (r1_num < 0) {
		throw std::runtime_error(std::string(range.ast_node->begin()) + ": Error: Can't convert range start " + r1 + " to a non-negative number");
	}

	if (!is_number(r2)) {
		throw std::runtime_error(std::string(range.ast_node->begin()) + ": Error: Can't convert range finish " + r2 + " to a non-negative number");
	}

	auto r2_num = std::stoi(r2);

	if (r2_num < 0) {
		throw std::runtime_error(std::string(range.ast_node->begin()) + ": Error: Can't convert range finish " + r2 + " to a non-negative number");
	}

	if (r1_num >= r2_num) {
		throw std::runtime_error(std::string(range.ast_node->begin()) + ": Error: start of the range " +
			r1 + " is greater or equal to finish " + r2);
	}

	current_test->cksum_input += "RANGE ";
	current_test->cksum_input += r1;
	current_test->cksum_input += r2;

	return range.values();
}

void VisitorSemantic::visit_for_clause(std::shared_ptr<AST::ForClause> for_clause) {
	current_test->cksum_input += "for ";
	std::vector<std::string> values;
	if (auto p = std::dynamic_pointer_cast<AST::CounterList<AST::Range>>(for_clause->counter_list)) {
		values = visit_range({p->counter_list, stack});
	} else {
		throw std::runtime_error("Unknown counter list type");
	}

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
		current_test->cksum_input += "else ";
		visit_action(for_clause->else_action);
	}
}

void VisitorSemantic::visit_cycle_control(const IR::CycleControl& cycle_control) {
	current_test->cksum_input += cycle_control.type();
}

void VisitorSemantic::visit_machine(std::shared_ptr<IR::Machine> machine) {
	current_test->mentioned_machines.insert(machine);

	auto result = visited_machines.insert(machine);
	if (!result.second) {
		return;
	}

	StackPusher<VisitorSemantic> new_ctx(this, machine->stack);

	machine->config = visit_attr_block(machine->ast_node->attr_block, "vm_global");
	machine->config["prefix"] = prefix;
	machine->config["name"] = machine->name();
	machine->config["src_file"] = machine->ast_node->name.begin().file.generic_string();

	if (machine->config.count("iso")) {
		fs::path iso_file = machine->config.at("iso").get<std::string>();
		if (iso_file.is_relative()) {
			fs::path src_file(machine->config.at("src_file").get<std::string>());
			iso_file = src_file.parent_path() / iso_file;
		}

		if (!fs::exists(iso_file)) {
			throw std::runtime_error(fmt::format("Can't construct VmController for vm \"{}\": target iso file \"{}\" does not exist", machine->name(), iso_file.generic_string()));
		}

		iso_file = fs::canonical(iso_file);

		machine->config["iso"] = iso_file.generic_string();
	}

	if (machine->config.count("loader")) {
		fs::path loader_file = machine->config.at("loader").get<std::string>();
		if (loader_file.is_relative()) {
			fs::path src_file(machine->config.at("src_file").get<std::string>());
			loader_file = src_file.parent_path() / loader_file;
		}

		if (!fs::exists(loader_file)) {
			throw std::runtime_error(fmt::format("Can't construct VmController for vm \"{}\": target loader file \"{}\" does not exist", machine->name(), loader_file.generic_string()));
		}

		loader_file = fs::canonical(loader_file);

		machine->config["loader"] = loader_file.generic_string();
	}

	if (machine->config.count("disk")) {
		auto& disks = machine->config.at("disk");

		for (auto& disk: disks) {
			if (disk.count("source")) {
				fs::path source_file = disk.at("source").get<std::string>();
				if (source_file.is_relative()) {
					fs::path src_file(machine->config.at("src_file").get<std::string>());
					source_file = src_file.parent_path() / source_file;
				}

				if (!fs::exists(source_file)) {
					throw std::runtime_error(fmt::format("Can't construct VmController for vm \"{}\": source disk image \"{}\" does not exist", machine->name(), source_file.generic_string()));
				}

				source_file = fs::canonical(source_file);
				disk["source"] = source_file;
			}
		}
	}
}

void VisitorSemantic::visit_flash(std::shared_ptr<IR::FlashDrive> flash) {
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
	flash->config["src_file"] = flash->ast_node->name.begin().file.generic_string();

	if (flash->has_folder()) {
		flash->validate_folder();
	}
}

void VisitorSemantic::visit_network(std::shared_ptr<IR::Network> network) {
	current_test->mentioned_networks.insert(network);

	auto result = visited_networks.insert(network);
	if (!result.second) {
		return;
	}

	StackPusher<VisitorSemantic> new_ctx(this, network->stack);

	network->config = visit_attr_block(network->ast_node->attr_block, "network_global");
	network->config["prefix"] = prefix;
	network->config["name"] = network->name();
	network->config["src_file"] = network->ast_node->name.begin().file.generic_string();
}

nlohmann::json VisitorSemantic::visit_attr_block(std::shared_ptr<AST::AttrBlock> attr_block, const std::string& ctx_name) {
	nlohmann::json config;
	for (auto attr: attr_block->attrs) {
		visit_attr(attr, config, ctx_name);
	}
	return config;
}

void VisitorSemantic::visit_attr(std::shared_ptr<AST::Attr> attr, nlohmann::json& config, const std::string& ctx_name) {
	auto ctx = attr_ctxs.find(ctx_name);
	if (ctx == attr_ctxs.end()) {
		throw std::runtime_error("Unknown ctx"); //should never happen
	}

	auto found = ctx->second.find(attr->name);

	if (found == ctx->second.end()) {
		throw std::runtime_error(std::string(attr->begin()) + ": Error: unknown attribute name: \"" + attr->name.value() + "\"");
	}

	auto match = found->second;
	if (attr->id != match.first) {
		if (match.first) {
			throw std::runtime_error(std::string(attr->end()) + ": Error: attribute \"" + attr->name.value() +
				"\" requires a name");
		} else {
			throw std::runtime_error(std::string(attr->end()) + ": Error: attribute \"" + attr->name.value() +
				"\" must have no name");
		}
	}

	if (attr->value->t.type() != match.second) {
		throw std::runtime_error(std::string(attr->end()) + ": Error: unexpected value type \"" +
			Token::type_to_string(attr->value->t.type()) + "\" for attribute \"" + attr->name.value() + "\", expected \"" +
			Token::type_to_string(match.second) + "\"");
	}

	if (config.count(attr->name.value())) {
		if (!config.at(attr->name.value()).is_array()) {
			throw std::runtime_error(std::string(attr->begin()) + ": Error: duplicate attribute: \"" + attr->name.value() + "\"");
		}
	}

	if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::StringAttr>>(attr->value)) {
		try {
			auto value = template_parser.resolve(p->attr_value->value->text(), stack);
			config[attr->name.value()] = value;
		} catch (const std::exception& error) {
			std::throw_with_nested(ResolveException(p->attr_value->value->begin(), p->attr_value->value->text()));
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::BinaryAttr>>(attr->value)) {
		auto value = p->attr_value->value;
		if (value.type() == Token::category::true_) {
			config[attr->name.value()] = true;
		} else if (value.type() == Token::category::false_) {
			config[attr->name.value()] = false;
		} else {
			throw std::runtime_error(std::string(attr->begin()) + ": Error: unsupported binary attr: " + value.value());
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::SimpleAttr>>(attr->value)) {
		auto value = p->attr_value->t;
		if (value.type() == Token::category::number) {
			if (std::stoi(value.value()) < 0) {
				throw std::runtime_error(std::string(attr->begin()) + ": Error: numeric attr can't be negative: " + value.value());
			}
			config[attr->name.value()] = std::stoul(value.value());
		} else if (value.type() == Token::category::size) {
			config[attr->name.value()] = size_to_mb(value);
		} else {
			 throw std::runtime_error(std::string(attr->begin()) + ": Error: unsupported attr: " + value.value());
		}
	} else if (auto p = std::dynamic_pointer_cast<AST::AttrValue<AST::AttrBlock>>(attr->value)) {
		//we assume for now that named attrs could be only in attr_blocks
		auto j = visit_attr_block(p->attr_value, attr->name);
		if (attr->id) {
			j["name"] = attr->id.value();
			config[attr->name.value()].push_back(j);
		}  else {
			config[attr->name.value()] = visit_attr_block(p->attr_value, attr->name);
		}
	} else {
		throw std::runtime_error("Unknown attr category");
	}
}

bool VisitorSemantic::is_button(const Token& t) const {
	std::string button = t.value();
	std::transform(button.begin(), button.end(), button.begin(), ::toupper);
	return (keys.find(button) != keys.end());
}
