
#include "../NNClient.hpp"
#include "../backends/Environment.hpp"
#include "VisitorSemantic.hpp"
#include "../IR/Program.hpp"
#include "../Exceptions.hpp"
#include "../Logger.hpp"
#include <fmt/format.h>
#include <wildcards.hpp>

struct ControllerCreatonException: public Exception {
	ControllerCreatonException(std::shared_ptr<IR::Controller> controller): Exception({}) {
		std::stringstream ss;
		ss << controller->macro_call_stack << std::string(controller->ast_node->begin())
			<< ": In the " << controller->type() << " \"" << controller->name() << "\" declaration";
		msg = ss.str();
	}
};

VisitorSemantic::VisitorSemantic(const VisitorSemanticConfig& config) {
	TRACE();
	prefix = config.prefix;
}

VisitorSemantic::~VisitorSemantic() {
	TRACE();
}

void VisitorSemantic::visit() {
	TRACE();

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
					throw ExceptionWithPos(test->ast_node->begin(), "Error: some parents have common virtual machines");
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
					throw ExceptionWithPos(test->ast_node->begin(), "Error: some parents have common flash drives");
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
			test->attrs = IR::AttrBlock(test->ast_node->attrs, stack).to_json();
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
			ss << test->macro_call_stack;
			std::string msg = ss.str();
			std::throw_with_nested(Exception(msg.substr(0, msg.length() - 1)));
		} else {
			throw;
		}
	}
}

void VisitorSemantic::visit_command_block(std::shared_ptr<AST::Block<AST::Cmd>> block) {
	for (auto command: block->items) {
		visit_command(command);
	}
}

void VisitorSemantic::visit_command(std::shared_ptr<AST::Cmd> cmd) {
	if (auto p = std::dynamic_pointer_cast<AST::RegularCmd>(cmd)) {
		visit_regular_command({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MacroCall<AST::Cmd>>(cmd)) {
		visit_cmd_macro_call({p, stack});
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
		throw ExceptionWithPos(regular_cmd.ast_node->entity->begin(), "Error: unknown virtual entity: " + regular_cmd.entity());
	}
	current_test->cksum_input << "}" << std::endl;

	current_controller = nullptr;
}

void VisitorSemantic::visit_action_block(std::shared_ptr<AST::Block<AST::Action>> action_block) {
	for (auto action: action_block->items) {
		visit_action(action);
	}
}

void VisitorSemantic::visit_action(std::shared_ptr<AST::Action> action) {
	if (std::dynamic_pointer_cast<IR::Machine>(current_controller)) {
		visit_action_vm(action);
	} else {
		visit_action_fd(action);
	}
}

void VisitorSemantic::visit_action_vm(std::shared_ptr<AST::Action> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Abort>(action)) {
		visit_abort({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::ActionWithDelim>(action)) {
		visit_action_vm(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Print>(action)) {
		visit_print({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::REPL>(action)) {
		visit_repl({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Type>(action)) {
		visit_type({p, stack, nullptr});
	} else if (auto p = std::dynamic_pointer_cast<AST::Press>(action)) {
		visit_press({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Hold>(action)) {
		visit_hold({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Release>(action)) {
		visit_release({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Block<AST::Action>>(action)) {
		visit_action_block(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::Mouse>(action)) {
		visit_mouse({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Plug>(action)) {
		visit_plug({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Start>(action)) {
		visit_start({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Stop>(action)) {
		visit_stop({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Shutdown>(action)) {
		visit_shutdown({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Exec>(action)) {
		visit_exec({p, stack, nullptr});
	} else if (auto p = std::dynamic_pointer_cast<AST::Copy>(action)) {
		visit_copy({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Screenshot>(action)) {
		visit_screenshot({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Wait>(action)) {
		visit_wait({p, stack, nullptr});
	} else if (auto p = std::dynamic_pointer_cast<AST::Sleep>(action)) {
		visit_sleep({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MacroCall<AST::Action>>(action)) {
		visit_action_macro_call({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::IfClause>(action)) {
		visit_if_clause(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::ForClause>(action)) {
		visit_for_clause(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::CycleControl>(action)) {
		visit_cycle_control({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Empty>(action)) {
		// do nothing
	} else {
		throw ExceptionWithPos(action->begin(), "Error: The action \"" + action->to_string() + "\" is not applicable to a virtual machine");
	}
}

void VisitorSemantic::visit_action_fd(std::shared_ptr<AST::Action> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Abort>(action)) {
		visit_abort({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::ActionWithDelim>(action)) {
		visit_action_fd(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Print>(action)) {
		visit_print({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::REPL>(action)) {
		visit_repl({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Copy>(action)) {
		visit_copy({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Sleep>(action)) {
		visit_sleep({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Block<AST::Action>>(action)) {
		visit_action_block(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::Empty>(action)) {
		// do nothing
	} else if (auto p = std::dynamic_pointer_cast<AST::MacroCall<AST::Action>>(action)) {
		visit_action_macro_call({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::IfClause>(action)) {
		visit_if_clause(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::ForClause>(action)) {
		visit_for_clause(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::CycleControl>(action)) {
		visit_cycle_control({p, stack});
	} else {
		throw ExceptionWithPos(action->begin(), "Error: The action \"" + action->to_string() + "\" is not applicable to a flash drive");
	}
}

void VisitorSemantic::visit_abort(const IR::Abort& abort) {
	current_test->cksum_input << "abort \"" << abort.message() << "\"" << std::endl;
}

void VisitorSemantic::visit_print(const IR::Print& print) {
	current_test->cksum_input << "print \"" << print.message() << "\"" << std::endl;
}

void VisitorSemantic::visit_repl(const IR::REPL& repl) {
	current_test->cksum_input << "repl \"" << rand() << "\"" << std::endl;
}

void VisitorSemantic::visit_type(const IR::Type& type) {
	type.validate();

	current_test->cksum_input << "type "
		<< "\"" << type.text().str() << "\""
		<< " interval " << type.interval().value().count();

	if (type.use_autoswitch()) {
		current_test->cksum_input << " autoswitch ";
		visit_key_combination(type.autoswitch());
	}
	current_test->cksum_input << std::endl;
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

	current_test->cksum_input << " interval " << press.interval().value().count() << std::endl;
}

void VisitorSemantic::visit_key_combination(const IR::KeyCombination& combination) {
	auto buttons = combination.buttons();
	for (size_t i = 0; i < buttons.size(); ++i) {
		auto button = buttons[i];

		for (size_t j = i + 1; j < buttons.size(); ++j) {
			if (button == buttons[j]) {
				throw ExceptionWithPos(combination.get_parsed()->buttons[j].begin(),
					"Error: duplicate key: " + combination.get_parsed()->buttons[j].value());
			}
		}

		if (i) {
			current_test->cksum_input << "+";
		}
		current_test->cksum_input << ToString(button);
	}
}

void VisitorSemantic::visit_key_spec(const IR::KeySpec& key_spec) {
	visit_key_combination(key_spec.combination());

	auto times = key_spec.times();

	if (times < 1) {
		throw ExceptionWithPos(key_spec.ast_node->times->begin(),
			"Error: can't press a button less than 1 time: " + std::to_string(times));
	}

	current_test->cksum_input << "*" << times;
}

void VisitorSemantic::visit_hold(const IR::Hold& hold) {
	current_test->cksum_input << "hold ";
	visit_key_combination(hold.combination());
	current_test->cksum_input << std::endl;
}

void VisitorSemantic::visit_release(const IR::Release& release) {
	current_test->cksum_input << "release";
	if (release.ast_node->combination) {
		current_test->cksum_input << " ";
		visit_key_combination(release.combination());
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
		current_test->cksum_input << specifier->to_string();
		if (specifier->is_from()) {
			if (!specifier->arg) {
				throw ExceptionWithPos(specifier->begin(), "Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}

			IR::Number arg(specifier->arg, stack);
			if (arg.value() < 0) {
				throw ExceptionWithPos(specifier->begin(), "Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}

			if (has_from) {
				throw ExceptionWithPos(specifier->begin(), "Error: you can't use specifier " + specifier->name.value() + " after another \"from\" specifier");
			}
			if (has_center) {
				throw ExceptionWithPos(specifier->begin(), "Error: you can't use specifier " + specifier->name.value() + " after a \"precision\" specifier");
			}
			if (has_move) {
				throw ExceptionWithPos(specifier->begin(), "Error: you can't use specifier " + specifier->name.value() + " after a \"move\" specifier");
			}
			has_from = true;
			continue;
		} if (specifier->is_centering()) {
			if (specifier->arg) {
				throw ExceptionWithPos(specifier->begin(), "Error: specifier " + specifier->name.value() + " must not have an argument");
			}
			if (has_center) {
				throw ExceptionWithPos(specifier->begin(), "Error: you can't use specifier " + specifier->name.value() + " after another \"precision\" specifier");
			}
			if (has_move) {
				throw ExceptionWithPos(specifier->begin(), "Error: you can't use specifier " + specifier->name.value() + " after a \"move\" specifier");
			}
			has_center = true;
			continue;
		} else if (specifier->is_moving()) {
			if (!specifier->arg) {
				throw ExceptionWithPos(specifier->begin(), "Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}

			IR::Number arg(specifier->arg, stack);
			if (arg.value() < 0) {
				throw ExceptionWithPos(specifier->begin(), "Error: specifier " + specifier->name.value() + " requires a non-negative number as an argument");
			}
			has_move = true;
			continue;
		} else {
			throw ExceptionWithPos(specifier->begin(), "Error: unknown specifier: " + specifier->name.value());
		}

	}
}

void VisitorSemantic::visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates) {
	if (coordinates.x_is_relative() ^ coordinates.y_is_relative()) {
		throw ExceptionWithPos(coordinates.ast_node->begin(), "Error: mouse coordinates must be either both absolute either both relative");
	}

	current_test->cksum_input << coordinates.x() << " " << coordinates.y();
}

void VisitorSemantic::validate_js(const std::string& js_script) {
	auto script = fmt::format("function __testo__() {{\n{}\n}}\nlet result = __testo__()\nJSON.stringify(result)", js_script);

	try {
		js_ctx.compile(script);
	} catch (const std::exception& error) {
		throw std::runtime_error(error.what());
	}
}

void VisitorSemantic::visit_select_js(const IR::SelectJS& js) {
	auto script = js.script();

	if (!script.length()) {
		throw ExceptionWithPos(js.ast_node->begin(), "Error: empty script in js selection");
	}

	try {
		validate_js(script);
	} catch (const std::exception& error) {
		std::throw_with_nested(ExceptionWithPos(js.ast_node->begin(), "Error while validating js selection"));
	}

	current_test->cksum_input << "js \"" << script << "\"";
}

void VisitorSemantic::visit_select_img(const IR::SelectImg& select) {
	select.img().validate();

	current_test->cksum_input
		<< "img \"" << select.img().signature();
}

void VisitorSemantic::visit_select_text(const IR::SelectText& text) {
	auto txt = text.text();
	if (!txt.length()) {
		throw ExceptionWithPos(text.ast_node->begin(), "Error: empty string in text selection");
	}
	if (std::find(txt.begin(), txt.end(), '\n') != txt.end()) {
		throw ExceptionWithPos(text.ast_node->begin(), "Error: multiline strings are not supported in wait action");
	}

	current_test->cksum_input << "text \"" << txt << "\"";
}

void VisitorSemantic::visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable) {
	if (auto p = std::dynamic_pointer_cast<AST::SelectJS>(mouse_selectable.ast_node->basic_select_expr)) {
		if (mouse_selectable.ast_node->mouse_additional_specifiers.size()) {
			throw ExceptionWithPos(mouse_selectable.ast_node->mouse_additional_specifiers[0]->begin(), "Error: mouse specifiers are not supported for js selections");
		}
		visit_select_js({p, stack, nullptr});
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectText>(mouse_selectable.ast_node->basic_select_expr)) {
		visit_select_text({p, stack, nullptr});
		visit_mouse_additional_specifiers(mouse_selectable.ast_node->mouse_additional_specifiers);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectImg>(mouse_selectable.ast_node->basic_select_expr)) {
		visit_select_img({p, stack, nullptr});
		visit_mouse_additional_specifiers(mouse_selectable.ast_node->mouse_additional_specifiers);
	}
}

void VisitorSemantic::visit_mouse_move_click(const IR::MouseMoveClick& mouse_move_click) {
	current_test->cksum_input << mouse_move_click.event_type();
	if (mouse_move_click.ast_node->object) {
		current_test->cksum_input << " ";
		if (auto p = std::dynamic_pointer_cast<AST::MouseCoordinates>(mouse_move_click.ast_node->object)) {
			visit_mouse_move_coordinates({p, stack});
		} else if (auto p = std::dynamic_pointer_cast<AST::MouseSelectable>(mouse_move_click.ast_node->object)) {
			visit_mouse_move_selectable({p, stack, nullptr});
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

	if (auto p = std::dynamic_pointer_cast<AST::MouseMoveClick>(mouse.ast_node->event)) {
		visit_mouse_move_click({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseHold>(mouse.ast_node->event)) {
		visit_mouse_hold({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseRelease>(mouse.ast_node->event)) {
		visit_mouse_release({p, stack});
	}

	current_test->cksum_input << std::endl;
}

void VisitorSemantic::visit_plug(const IR::Plug& plug) {
	if (plug.is_on()) {
		current_test->cksum_input << "plug ";
	} else {
		current_test->cksum_input << "unplug ";
	}

	if (auto p = std::dynamic_pointer_cast<AST::PlugFlash>(plug.ast_node->resource)) {
		visit_plug_flash({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::PlugDVD>(plug.ast_node->resource)) {
		visit_plug_dvd({p, stack}, plug.is_on());
	} else if (auto p = std::dynamic_pointer_cast<AST::PlugNIC>(plug.ast_node->resource)) {
		visit_plug_nic({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::PlugLink>(plug.ast_node->resource)) {
		visit_plug_link({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::PlugHostDev>(plug.ast_node->resource)) {
		visit_plug_hostdev({p, stack});
	} else {
		throw Exception("unknown hardware to plug/unplug: " +
			plug.ast_node->resource->to_string());
	}

	current_test->cksum_input << std::endl;
}

void VisitorSemantic::visit_plug_flash(const IR::PlugFlash& plug_flash) {
	current_test->cksum_input << "flash " << plug_flash.name();

	auto flash_drive = IR::program->get_flash_drive_or_null(plug_flash.name());
	if (!flash_drive) {
		throw ExceptionWithPos(plug_flash.ast_node->begin(), "Error: unknown flash drive: " + plug_flash.name());
	}
	visit_flash(flash_drive);

}

void VisitorSemantic::visit_plug_dvd(const IR::PlugDVD& plug_dvd, bool is_on) {
	current_test->cksum_input << "dvd";

	if (is_on) {
		auto dvd_path = plug_dvd.path();
		if (!fs::exists(dvd_path)) {
			throw ExceptionWithPos(plug_dvd.ast_node->begin(), "Error: specified dvd image path does not exist: " + dvd_path.generic_string());
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
		throw ExceptionWithPos(plug_hostdev.ast_node->begin(), "Sorry, Hyper-V does not support this command");
	}

	try {
		parse_usb_addr(plug_hostdev.addr());
	} catch (const std::exception& error) {
		throw ExceptionWithPos(plug_hostdev.ast_node->begin(), "Error: spicified usb addr is not valid: " + plug_hostdev.addr());
	}
}

void VisitorSemantic::visit_start(const IR::Start& start) {
	current_test->cksum_input << "start" << std::endl;
}

void VisitorSemantic::visit_stop(const IR::Stop& stop) {
	current_test->cksum_input << "stop" << std::endl;
}

void VisitorSemantic::visit_shutdown(const IR::Shutdown& shutdown) {
	current_test->cksum_input << "shutdown timeout " << shutdown.timeout().value().count() << std::endl;
}

void VisitorSemantic::visit_exec(const IR::Exec& exec) {
	if ((exec.interpreter() != "bash") &&
		(exec.interpreter() != "cmd") &&
		(exec.interpreter() != "python") &&
		(exec.interpreter() != "python2") &&
		(exec.interpreter() != "python3"))
	{
		throw ExceptionWithPos(exec.ast_node->begin(), "Error: unknown process name: " + exec.interpreter());
	}

	current_test->cksum_input << "exec "
		<< exec.interpreter() << " \"\"\"" << exec.script() << "\"\"\""
		<< " timeout " << exec.timeout().value().count()
		<< std::endl;
}

void VisitorSemantic::visit_copy(const IR::Copy& copy) {
	if (copy.ast_node->is_to_guest()) {
		current_test->cksum_input << "copyto ";
	} else {
		current_test->cksum_input << "copyfrom ";
	}

	current_test->cksum_input << copy.from()<< " " << copy.to() << " timeout " << copy.timeout().value().count() << std::endl;

	auto from = copy.from();
	if (copy.ast_node->is_to_guest()) {

		if (!copy.nocheck()) {
			if (!fs::exists(from)) {
				throw ExceptionWithPos(copy.ast_node->begin(), "Error: specified path doesn't exist: " + from);
			}

			current_test->cksum_input << pretty_files_signature(from) << std::endl;
		} else {
			current_test->cksum_input << " nocheck" << std::endl;
		}

	} else {
		if (copy.nocheck()) {
			throw ExceptionWithPos(copy.ast_node->begin(), "Error: \"nocheck\" specifier is not applicable to copyfrom action");
		}
	}
}

void VisitorSemantic::visit_screenshot(const IR::Screenshot& screenshot) {
	auto destination = screenshot.destination();
	current_test->cksum_input << "screenshot" << " " << destination << std::endl;

	//No additional checks needed
}

void VisitorSemantic::visit_detect_expr(std::shared_ptr<AST::SelectExpr> select_expr) {
	if (auto p = std::dynamic_pointer_cast<AST::SelectNegationExpr>(select_expr)) {
		current_test->cksum_input << "!";
		visit_detect_expr(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectText>(select_expr)) {
		visit_select_text({p, stack, nullptr});
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectJS>(select_expr)) {
		visit_select_js({p, stack, nullptr});
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectImg>(select_expr)) {
		visit_select_img({p, stack, nullptr});
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectParentedExpr>(select_expr)) {
		visit_detect_parented(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectBinOp>(select_expr)) {
		visit_detect_binop(p);
	} else {
		throw Exception("Unknown detect expr type");
	}
}

void VisitorSemantic::visit_detect_parented(std::shared_ptr<AST::SelectParentedExpr> parented) {
	current_test->cksum_input << "(";
	visit_detect_expr(parented->select_expr);
	current_test->cksum_input << ")";

}

void VisitorSemantic::visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop) {
	visit_detect_expr(binop->left);
	current_test->cksum_input << binop->op.value();
	visit_detect_expr(binop->right);
}

void VisitorSemantic::visit_wait(const IR::Wait& wait) {
	current_test->cksum_input << "wait ";
	visit_detect_expr(wait.ast_node->select_expr);
	current_test->cksum_input << " timeout " << wait.timeout().value().count()
		<< " interval " << wait.interval().value().count()
		<< std::endl;
}

void VisitorSemantic::visit_sleep(const IR::Sleep& sleep) {
	current_test->cksum_input << "sleep timeout " << sleep.timeout().value().count() << std::endl;
}

void VisitorSemantic::visit_cmd_macro_call(const IR::MacroCall& macro_call) {
	macro_call.visit_semantic<AST::Cmd>(this);
}

void VisitorSemantic::visit_action_macro_call(const IR::MacroCall& macro_call) {
	macro_call.visit_semantic<AST::Action>(this);
}

void VisitorSemantic::visit_macro_body(const std::shared_ptr<AST::Block<AST::Cmd>>& macro_body) {
	visit_command_block(macro_body);
}

void VisitorSemantic::visit_macro_body(const std::shared_ptr<AST::Block<AST::Action>>& macro_body) {
	visit_action_block(macro_body);
}


Tribool VisitorSemantic::visit_expr(std::shared_ptr<AST::Expr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::BinOp>(expr)) {
		return visit_binop(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::Negation>(expr)) {
		current_test->cksum_input << "NOT ";
		return !visit_expr(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Check>(expr)) {
		return visit_check({ p, stack, nullptr });
	} else if (auto p = std::dynamic_pointer_cast<AST::Defined>(expr)) {
		return visit_defined({ p, stack });
	} else if (auto p = std::dynamic_pointer_cast<AST::Comparison>(expr)) {
		return visit_comparison({ p, stack });
	} else if (auto p = std::dynamic_pointer_cast<AST::ParentedExpr>(expr)) {
		return visit_parented_expr(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::StringExpr>(expr)) {
		auto text = IR::String(p->str, stack).text();
		current_test->cksum_input << "\"" << text << "\"";
		return text.length() ? Tribool::yes : Tribool::no;
	} else {
		throw Exception("Unknown expr type");
	}
}


Tribool VisitorSemantic::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	auto left = visit_expr(binop->left);
	current_test->cksum_input << " " << binop->op.value() << " ";

	if (binop->op.value() == "AND") {
		if (left == Tribool::no) {
			return left;
		} else {
			return visit_expr(binop->right);
		}
	} else if (binop->op.value() == "OR") {
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

Tribool VisitorSemantic::visit_parented_expr(std::shared_ptr<AST::ParentedExpr> parented) {
	current_test->cksum_input << "(";
	auto result = visit_expr(parented->expr);
	current_test->cksum_input << ")";
	return result;
}

Tribool VisitorSemantic::visit_check(const IR::Check& check) {
	if (std::dynamic_pointer_cast<IR::FlashDrive>(current_controller)) {
		throw ExceptionWithPos(check.ast_node->begin(), "Error: The \"check\" expression is not applicable to a flash drive");
	}

	current_test->cksum_input << "check ";
	visit_detect_expr(check.ast_node->select_expr);
	current_test->cksum_input
		<< " timeout " << check.timeout().value().count()
		<< " interval " << check.interval().value().count();
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
	int32_t r1 = range.r1();
	int32_t r2 = range.r2();

	if (r1 < 0) {
		throw ExceptionWithPos(range.ast_node->begin(), "Error: Can't convert range start " + std::to_string(r1) + " to a non-negative number");
	}

	if (r2 < 0) {
		throw ExceptionWithPos(range.ast_node->begin(), "Error: Can't convert range finish " + std::to_string(r2) + " to a non-negative number");
	}

	if (r1 >= r2) {
		throw ExceptionWithPos(range.ast_node->begin(), "Error: start of the range " +
			std::to_string(r1) + " is greater or equal to finish " + std::to_string(r2));
	}

	return range.values();
}

void VisitorSemantic::visit_for_clause(std::shared_ptr<AST::ForClause> for_clause) {
	current_test->cksum_input << "for (";
	std::vector<std::string> values;
	if (auto p = std::dynamic_pointer_cast<AST::Range>(for_clause->counter_list)) {
		values = visit_range({p, stack});
	} else {
		throw Exception("Unknown counter list type");
	}
	current_test->cksum_input << ") {" << std::endl;

	std::map<std::string, std::string> params;
	for (auto i: values) {
		params[for_clause->counter.value()] = i;
		auto new_stack = std::make_shared<StackNode>();
		new_stack->parent = stack;
		new_stack->params = params;
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

		machine->config = IR::AttrBlock(machine->ast_node->attr_block, stack).to_json();
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
		flash->config = IR::AttrBlock(flash->ast_node->attr_block, stack).to_json();
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

		network->config = IR::AttrBlock(network->ast_node->attr_block, stack).to_json();
		network->config["prefix"] = prefix;
		network->config["name"] = network->name();
		network->config["src_file"] = network->ast_node->name->begin().file.generic_string();

		network->validate_config();
	} catch (const std::exception& error) {
		std::throw_with_nested(ControllerCreatonException(network));
	}
}
