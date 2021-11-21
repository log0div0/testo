
#include <coro/CheckPoint.h>
#include <coro/Timeout.h>
#include "VisitorInterpreterActionMachine.hpp"
#include "NNClient.hpp"
#include "Exceptions.hpp"
#include "backends/Environment.hpp"
#include "IR/Program.hpp"
#include <fmt/format.h>

using namespace std::chrono_literals;

static std::string escape_text(const std::string& text) {
	std::string final_text;

	for (auto i: text) {
		if (i == '"') {
			final_text += '\\';
		}

		if (i == '\\') {
			final_text += '\\';
		}

		final_text += i;
	}

	return final_text;
}

static std::string build_shell_script(const std::string& body) {
	std::string script = "set -e; set -o pipefail; set -x;";
	script += body;
	script.erase(std::remove(script.begin(), script.end(), '\r'), script.end());

	return script;
}

static std::string build_batch_script(const std::string& body) {
	std::string script = "chcp 65001\n";
	script += body;
	return script;
}

static std::string build_python_script(const std::string& body) {
	std::vector<std::string> strings;
	std::stringstream iss(body);
	while(iss.good())
	{
		std::string single_string;
		getline(iss,single_string,'\n');
		strings.push_back(single_string);
	}

	size_t base_offset = 0;

	bool offset_found = false;

	for (auto& str: strings) {
		size_t offset_probe = 0;

		if (offset_found) {
			break;
		}

		for (auto it = str.begin(); it != str.end(); ++it) {
			while (*it == '\t') {
				offset_probe++;
				++it;
			}
			if (it == str.end()) {
				//empty string
				break;
			} else {
				//meaningful_string
				base_offset = offset_probe;
				offset_found = true;
				break;
			}
		}
	}

	std::string result;

	for (auto& str: strings) {
		for (auto it = str.begin(); it != str.end(); ++it) {
			for (size_t i = 0; i < base_offset; i++) {
				if (it == str.end()) {
					break;
				}

				if (*it != '\t') {
					throw std::runtime_error("Ill-formatted python script");
				}
				++it;
			}

			result += std::string(it, str.end());
			result += "\n";
			break;
		}
	}

	return result;
}

VisitorInterpreterActionMachine::VisitorInterpreterActionMachine(std::shared_ptr<IR::Machine> vmc, std::shared_ptr<StackNode> stack, Reporter& reporter, std::shared_ptr<IR::Test> current_test):
	VisitorInterpreterAction(vmc, stack, reporter), vmc(vmc), current_test(current_test)
{

}

void VisitorInterpreterActionMachine::visit_action(std::shared_ptr<AST::Action> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Abort>(action)) {
		visit_abort({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::ActionWithDelim>(action)) {
		visit_action(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Print>(action)) {
		visit_print({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Type>(action)) {
		visit_type({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Wait>(action)) {
		visit_wait({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Sleep>(action)) {
		visit_sleep({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Press>(action)) {
		visit_press({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Hold>(action)) {
		visit_hold({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Release>(action)) {
		visit_release({p, stack});
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
		visit_exec({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Copy>(action)) {
		visit_copy({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Screenshot>(action)) {
		visit_screenshot({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MacroCall<AST::Action>>(action)) {
		visit_macro_call({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::IfClause>(action)) {
		visit_if_clause(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::ForClause>(action)) {
		visit_for_clause(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::CycleControl>(action)) {
		throw CycleControlException(p->token);
	} else if (auto p = std::dynamic_pointer_cast<AST::Block<AST::Action>>(action)) {
		visit_action_block(p);
	} else if (auto p = std::dynamic_pointer_cast<AST::Empty>(action)) {
		;
	} else {
		throw std::runtime_error("Should never happen");
	}

	coro::CheckPoint();
}

void VisitorInterpreterActionMachine::visit_copy(const IR::Copy& copy) {
	try {
		reporter.copy(current_controller, copy);

		coro::Timeout timeout(copy.timeout().value());

		if (vmc->vm()->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		auto ga = vmc->vm()->guest_additions();

		if (!ga->is_avaliable()) {
			throw std::runtime_error(fmt::format("guest additions are not installed"));
		}

		if(copy.ast_node->is_to_guest()) {
			//Additional check since now we can't be sure the "from" actually exists
			if (!fs::exists(copy.from())) {
				throw std::runtime_error("Specified path doesn't exist: " + copy.from());
			}
			ga->copy_to_guest(copy.from(), copy.to());
		} else {
			ga->copy_from_guest(copy.from(), copy.to());;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(copy.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_screenshot(const IR::Screenshot& screenshot) {
	try {
		fs::path destination = screenshot.destination();
		reporter.screenshot(vmc, screenshot);

		if (vmc->vm()->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		auto& screenshot = vmc->make_new_screenshot();

		if (!fs::exists(destination.parent_path())) {
			if (!fs::create_directories(destination.parent_path())) {
				throw std::runtime_error("Can't create directory: " + destination.parent_path().generic_string());
			}
		}

		screenshot.write_png(destination.generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(screenshot.ast_node, current_controller));
	}
}

bool VisitorInterpreterActionMachine::visit_check(const IR::Check& check) {
	try {
		reporter.check(vmc, check);

		return screenshot_loop([&](const stb::Image<stb::RGB>& screenshot) {
			return visit_detect_expr(check.ast_node->select_expr, screenshot);
		}, check.timeout().value(), check.interval().value());

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(check.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_key_combination(const IR::KeyCombination& key_combination, std::chrono::milliseconds interval) {
	std::vector<KeyboardButton> buttons = key_combination.buttons();
	for (auto it = buttons.begin(); it != buttons.end(); ++it) {
		vmc->hold(*it);
	}
	for (auto it = buttons.rbegin(); it != buttons.rend(); ++it) {
		vmc->release(*it);
	}
	timer.waitFor(interval);
}

void VisitorInterpreterActionMachine::execute_keyboard_commands(const std::vector<KeyboardCommand>& commands, std::chrono::milliseconds interval) {
	for (size_t i = 0; i < commands.size(); ++i) {
		if (i) {
			if ((commands[i-1].action == KeyboardAction::Release) &&
				(commands[i].action == KeyboardAction::Hold))
			{
				timer.waitFor(interval);
			}
		}
		switch (commands[i].action) {
			case KeyboardAction::Hold:
				vmc->hold(commands[i].button);
				break;
			case KeyboardAction::Release:
				vmc->release(commands[i].button);
				break;
			default:
				throw std::runtime_error("Should not be there");
		}
	}
	if (commands.size()) {
		timer.waitFor(interval);
	}
}

size_t VisitorInterpreterActionMachine::get_number_of(const std::string& text) {
	auto& screenshot = vmc->make_new_screenshot();

	if (!screenshot.data) {
		throw std::runtime_error("Failed to make a screenshot of the VM");
	}

	nlohmann::json json = eval_js("return find_text(\"" + escape_text(text) + "\").size()", screenshot);
	return json.get<size_t>();
}

struct LayoutSwitchCounter {
	void increment() {
		if (!tries) {
			tries.reset(new size_t(0));
		}
		++*tries;
		if (*tries > 10) {
			throw std::runtime_error("Failed to switch the keyboard layout automatically");
		}
	}
	void reset() {
		tries.reset();
	}
	std::unique_ptr<size_t> tries;
};

void VisitorInterpreterActionMachine::visit_type(const IR::Type& type) {
	try {
		type.validate();

		std::string text = type.text().str();
		if (text.size() == 0) {
			return;
		}

		auto interval = type.interval().value();
		reporter.type(vmc, type);
		std::vector<TypingPlan> chunks = KeyboardLayout::build_typing_plan(text);
		LayoutSwitchCounter counter;

		for (size_t j = 0; j < chunks.size();) {
			const TypingPlan& chunk = chunks[j];

			if (type.use_autoswitch() && (chunk.what_to_search().size() != 0)) {
				size_t before = get_number_of(chunk.what_to_search());
				execute_keyboard_commands(chunk.start_typing(), interval);
				size_t after = get_number_of(chunk.what_to_search());
				if (!(before < after)) {
					counter.increment();
					execute_keyboard_commands(chunk.rollback(), interval);
					visit_key_combination(type.autoswitch(), interval);
					continue;
				}
				counter.reset();
				execute_keyboard_commands(chunk.finish_typing(), interval);
				if (j != (chunks.size() - 1)) {
					visit_key_combination(type.autoswitch(), interval);
				}
			} else {
				execute_keyboard_commands(chunk.just_type_final_text(), interval);
			}

			++j;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(type.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_wait(const IR::Wait& wait) {
	try {
		reporter.wait(vmc, wait);

		bool early_exit = screenshot_loop([&](const stb::Image<stb::RGB>& screenshot) {
			return visit_detect_expr(wait.ast_node->select_expr, screenshot);
		}, wait.timeout().value(), wait.interval().value());

		if (!early_exit) {
			reporter.save_screenshot(vmc, vmc->get_last_screenshot());
			throw std::runtime_error("Timeout");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(wait.ast_node, current_controller));
	}
}

std::string VisitorInterpreterActionMachine::visit_mouse_specifier_from(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier) {
	auto name = specifier->name.value();
	auto arg = std::stoul(specifier->arg.value()); //should never fail since we have semantic checks

	if (name == "from_top") {
		return fmt::format(".from_top({})", arg);
	} else if (name == "from_bottom") {
		return fmt::format(".from_bottom({})", arg);
	} else if (name == "from_left") {
		return fmt::format(".from_left({})", arg);
	} else if (name == "from_right") {
		return fmt::format(".from_right({})", arg);
	}

	throw std::runtime_error("Should not be there");
}

std::string VisitorInterpreterActionMachine::visit_mouse_specifier_centering(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier) {
	auto name = specifier->name.value();

	if (name == "left_bottom") {
		return ".left_bottom()";
	} else if (name == "left_center") {
		return ".left_center()";
	} else if (name == "left_top") {
		return ".left_top()";
	} else if (name == "center_bottom") {
		return ".center_bottom()";
	} else if (name == "center") {
		return ".center()";
	} else if (name == "center_top") {
		return ".center_top()";
	} else if (name == "right_bottom") {
		return ".right_bottom()";
	} else if (name == "right_center") {
		return ".right_center()";
	} else if (name == "right_top") {
		return ".right_top()";
	}

	throw std::runtime_error("Uknown center specifier");
}

std::string VisitorInterpreterActionMachine::visit_mouse_specifier_default_centering() {
	return ".center()";
}

std::string VisitorInterpreterActionMachine::visit_mouse_specifier_moving(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier) {
	auto name = specifier->name.value();
	auto arg = std::stoul(specifier->arg.value()); //should never fail since we have semantic checks

	if (name == "move_left") {
		return fmt::format(".move_left({})", arg);
	} else if (name == "move_right") {
		return fmt::format(".move_right({})", arg);
	} else if (name == "move_up") {
		return fmt::format(".move_up({})", arg);
	} else if (name == "move_down") {
		return fmt::format(".move_down({})", arg);
	}

	throw std::runtime_error("Should not be there");
}

std::string VisitorInterpreterActionMachine::visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers) {
	size_t index = 0;

	std::string result;

	if ((specifiers.size() > index) && specifiers[index]->is_from()) {
		result += visit_mouse_specifier_from(specifiers[index]);
		index++;
	}

	if (specifiers.size() > index && specifiers[index]->is_centering()) {
		result += visit_mouse_specifier_centering(specifiers[index]);
		index++;
	} else {
		result += visit_mouse_specifier_default_centering();
	}

	for (size_t i = index; i < specifiers.size(); ++i) {
		result += visit_mouse_specifier_moving(specifiers[i]);
	}

	return result;
}


std::string VisitorInterpreterActionMachine::build_select_text_script(const IR::SelectText& text) {
	auto text_to_find = text.text();
	std::string final_text = escape_text(text_to_find);
	std::string result = fmt::format("return find_text(\"{}\")", final_text);
	return result;
}

std::string VisitorInterpreterActionMachine::build_select_img_script(const IR::SelectImg& img) {
	std::string result = fmt::format("return find_img('{}')", img.img_path().generic_string());
	return result;
}


bool VisitorInterpreterActionMachine::visit_detect_js(const IR::SelectJS& js, const stb::Image<stb::RGB>& screenshot) {
	auto value = eval_js(js.script(), screenshot);

	try {
		if (value.is_boolean()) {
			return (bool)value;
		} else {
		 	throw std::runtime_error("Can't process return value type. We expect a single boolean");
		}
	} catch (const std::exception&) {
		std::throw_with_nested(Exception("Error while processing a response message from NN server:\n" + value.dump(4)));
	}
}

bool VisitorInterpreterActionMachine::VisitorInterpreterActionMachine::visit_detect_expr(std::shared_ptr<AST::SelectExpr> select_expr, const stb::Image<stb::RGB>& screenshot)  {
	std::string script;

	if (auto p = std::dynamic_pointer_cast<AST::SelectNegationExpr>(select_expr)) {
		return !visit_detect_expr(p->expr, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectText>(select_expr)) {
		script = build_select_text_script({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectJS>(select_expr)) {
		return visit_detect_js({p, stack}, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectImg>(select_expr)) {
		script = build_select_img_script({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectParentedExpr>(select_expr)) {
		return visit_detect_expr(p->select_expr, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectBinOp>(select_expr)) {
		return visit_detect_binop(p, screenshot);
	} else {
		throw std::runtime_error("Unknown select expression type");
	}

	auto eval_result = eval_js(script, screenshot);
	try {
		if (eval_result.is_array()) {
			return (bool)eval_result.size();
		} else if (eval_result.is_boolean()) {
			return (bool)eval_result;
		} else {
			throw std::runtime_error("Uknown js return type: we expect array or boolean");
		}
	} catch (const std::exception&) {
		std::throw_with_nested(Exception("Error while processing a response message from NN server:\n" + eval_result.dump(4)));
	}
}

bool VisitorInterpreterActionMachine::visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop, const stb::Image<stb::RGB>& screenshot) {
	auto left_value = visit_detect_expr(binop->left, screenshot);
	if (binop->op.type() == Token::category::double_ampersand) {
		if (!left_value) {
			return false;
		} else {
			return left_value && visit_detect_expr(binop->right, screenshot);
		}
	} else if (binop->op.type() == Token::category::double_vertical_bar) {
		if (left_value) {
			return true;
		} else {
			return left_value || visit_detect_expr(binop->right, screenshot);
		}
	} else {
		throw std::runtime_error("Unknown binop operation");
	}
}

void VisitorInterpreterActionMachine::visit_press(const IR::Press& press) {
	try {
		IR::TimeInterval interval = press.interval();

		for (auto key_spec_: press.ast_node->keys) {
			IR::KeySpec key_spec(key_spec_, stack);

			uint32_t times = key_spec.times();

			reporter.press_key(vmc, key_spec);

			for (uint32_t i = 0; i < times; i++) {
				visit_key_combination(key_spec.combination(), interval.value());
			}
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(press.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_hold(const IR::Hold& hold) {
	try {
		auto buttons = hold.combination().buttons();

		reporter.hold_key(vmc, hold);
		for (KeyboardButton button: buttons) {
			vmc->hold(button);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(hold.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_release(const IR::Release& release) {
	try {
		auto buttons = release.combination().buttons();

		if (buttons.size()) {
			reporter.release_key(vmc, release);
			for (KeyboardButton button: buttons) {
				vmc->release(button);
			}
		} else {
			reporter.release_key(vmc);
			vmc->release();
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(release.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable) {
	reporter.mouse_move_click_selectable(vmc, mouse_selectable);

	bool early_exit = screenshot_loop([&](const stb::Image<stb::RGB>& screenshot) {
		Point point;
		try {
			std::string script;
			if (auto p = std::dynamic_pointer_cast<AST::SelectJS>(mouse_selectable.ast_node->basic_select_expr)) {
				script = IR::SelectJS(p, stack).script();
			} else if (auto p = std::dynamic_pointer_cast<AST::SelectText>(mouse_selectable.ast_node->basic_select_expr)) {
				script = build_select_text_script({p, stack});
				script += visit_mouse_additional_specifiers(mouse_selectable.ast_node->mouse_additional_specifiers);
			} else if (auto p = std::dynamic_pointer_cast<AST::SelectImg>(mouse_selectable.ast_node->basic_select_expr)) {
				script = build_select_img_script({p, stack});
				script += visit_mouse_additional_specifiers(mouse_selectable.ast_node->mouse_additional_specifiers);
			}

			auto js_result = eval_js(script, screenshot);

			try {
				if (js_result.is_object() && !js_result.is_array()) {
					if (!js_result.count("x")) {
						throw std::runtime_error("Object doesn't have the x propery");
					}

					if (!js_result.count("y")) {
						throw std::runtime_error("Object doesn't have the y propery");
					}

					auto x = js_result.at("x").get<int32_t>();
					auto y = js_result.at("y").get<int32_t>();

					point.x = x;
					point.y = y;

				} else {
					throw std::runtime_error("Can't process return value type. We expect a single object");
				}
			} catch (const std::exception&) {
				std::throw_with_nested(Exception("Error while processing a response message from NN server:\n" + js_result.dump(4)));
			}

			vmc->vm()->mouse_move_abs(point.x, point.y);
			return true;
		} catch (const ContinueError&) {
			return false;
		}
	}, mouse_selectable.timeout().value(), 1s);

	if (!early_exit) {
		reporter.save_screenshot(vmc, vmc->get_last_screenshot());
		throw std::runtime_error("Timeout");
	}
}

void VisitorInterpreterActionMachine::visit_mouse(const IR::Mouse& mouse) {
	if (auto p = std::dynamic_pointer_cast<AST::MouseMoveClick>(mouse.ast_node->event)) {
		return visit_mouse_move_click({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseHold>(mouse.ast_node->event)) {
		return visit_mouse_hold({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseRelease>(mouse.ast_node->event)) {
		return visit_mouse_release({p, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseWheel>(mouse.ast_node->event)) {
		throw std::runtime_error("Not implemented yet");
	} else {
		throw std::runtime_error("Unknown mouse actions");
	}
}

void VisitorInterpreterActionMachine::visit_mouse_move_click(const IR::MouseMoveClick& mouse_move_click) {
	try {
		reporter.mouse_move_click(vmc, mouse_move_click);

		if (mouse_move_click.ast_node->object) {
			if (auto p = std::dynamic_pointer_cast<AST::MouseCoordinates>(mouse_move_click.ast_node->object)) {
				visit_mouse_move_coordinates({p, stack});
			} else if (auto p = std::dynamic_pointer_cast<AST::MouseSelectable>(mouse_move_click.ast_node->object)) {
				visit_mouse_move_selectable({p, stack});
			} else {
				throw std::runtime_error("Unknown mouse move target");
			}
		} else {
			reporter.mouse_no_object();
		}

		if (mouse_move_click.event_type() == "move") {
			return;
		}

		auto mouse_press = [&](MouseButton button) {
			vmc->mouse_hold(button);
			timer.waitFor(std::chrono::milliseconds(60));
			vmc->mouse_release();
		};

		if (mouse_move_click.event_type() == "click" || mouse_move_click.event_type() == "lclick") {
			mouse_press(MouseButton::Left);
		} else if (mouse_move_click.event_type() == "rclick") {
			mouse_press(MouseButton::Right);
		} else if (mouse_move_click.event_type() == "mclick") {
			mouse_press(MouseButton::Middle);
		} else if (mouse_move_click.event_type() == "dclick") {
			mouse_press(MouseButton::Left);
			timer.waitFor(std::chrono::milliseconds(60));
			mouse_press(MouseButton::Left);
		} else {
			throw std::runtime_error("Unsupported click type");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_move_click.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates) {
	reporter.mouse_move_click_coordinates(vmc, coordinates);
	if (coordinates.x_is_relative() && coordinates.y_is_relative()) {
		vmc->vm()->mouse_move_rel(std::stoi(coordinates.x()), std::stoi(coordinates.y()));
	} else if (!coordinates.x_is_relative() && !coordinates.y_is_relative()) {
		vmc->vm()->mouse_move_abs(std::stoul(coordinates.x()), std::stoul(coordinates.y()));
	} else {
		throw std::runtime_error("Should not be there");
	}
}

void VisitorInterpreterActionMachine::visit_mouse_hold(const IR::MouseHold& mouse_hold) {
	try {
		reporter.mouse_hold(vmc, mouse_hold);
		vmc->mouse_hold(mouse_hold.button());
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_hold.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_mouse_release(const IR::MouseRelease& mouse_release) {
	try {
		reporter.mouse_release(vmc);
		vmc->mouse_release();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_release.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_mouse_wheel(const IR::MouseWheel& mouse_wheel) {
	try {
		reporter.mouse_wheel(vmc, mouse_wheel);

		auto mouse_press = [&](MouseButton button) {
			vmc->mouse_hold(button);
			timer.waitFor(std::chrono::milliseconds(60));
			vmc->mouse_release();
		};

		if (mouse_wheel.direction() == "up") {
			mouse_press(MouseButton::WheelUp);
		} else if (mouse_wheel.direction() == "down") {
			mouse_press(MouseButton::WheelDown);
		} else {
			throw std::runtime_error("Unknown wheel direction");
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_wheel.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_plug(const IR::Plug& plug) {
	try {
		if (auto p = std::dynamic_pointer_cast<AST::PlugFlash>(plug.ast_node->resource)) {
			if (plug.is_on()) {
				return visit_plug_flash({p, stack});
			} else {
				return visit_unplug_flash({p, stack});
			}
		} else if (auto p = std::dynamic_pointer_cast<AST::PlugDVD>(plug.ast_node->resource)) {
			if (plug.is_on()) {
				return visit_plug_dvd({p, stack});
			} else {
				return visit_unplug_dvd({p, stack});
			}
		} else if (auto p = std::dynamic_pointer_cast<AST::PlugHostDev>(plug.ast_node->resource)) {
			if (plug.is_on()) {
				return visit_plug_hostdev({p, stack});
			} else {
				return visit_unplug_hostdev({p, stack});
			}
		} else if (auto p = std::dynamic_pointer_cast<AST::PlugNIC>(plug.ast_node->resource)) {
			return visit_plug_nic({p, stack}, plug.is_on());
		} else if (auto p = std::dynamic_pointer_cast<AST::PlugLink>(plug.ast_node->resource)) {
			return visit_plug_link({p, stack}, plug.is_on());
		} else {
			throw std::runtime_error("unknown hardware to plug/unplug: " +
				plug.ast_node->resource->to_string());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(plug.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_plug_nic(const IR::PlugNIC& plug_nic, bool is_on) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vmc while semantic analisys
	auto nic = plug_nic.name();

	reporter.plug(vmc, "nic", nic, is_on);

	auto nics = vmc->vm()->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("specified nic {} is not present in this virtual machine", nic));
	}

	if (vmc->vm()->state() != VmState::Stopped) {
		throw std::runtime_error(fmt::format("virtual machine is running, but must be stopped"));
	}

	if (vmc->is_nic_plugged(nic) == is_on) {
		if (is_on) {
			throw std::runtime_error(fmt::format("specified nic {} is already plugged in this virtual machine", nic));
		} else {
			throw std::runtime_error(fmt::format("specified nic {} is not unplugged from this virtual machine", nic));
		}
	}

	if (is_on) {
		vmc->plug_nic(nic);
	} else {
		vmc->unplug_nic(nic);
	}
}

void VisitorInterpreterActionMachine::visit_plug_link(const IR::PlugLink& plug_link, bool is_on) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vmc while semantic analisys

	auto nic = plug_link.name();

	reporter.plug(vmc, "link", nic, is_on);

	auto nics = vmc->vm()->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is not present in this virtual machine", nic));
	}

	if (!vmc->is_nic_plugged(nic)) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is unplugged, you must to plug it first", nic));
	}

	if (is_on == vmc->is_link_plugged(nic)) {
		if (is_on) {
			throw std::runtime_error(fmt::format("specified link {} is already plugged in this virtual machine", nic));
		} else {
			throw std::runtime_error(fmt::format("specified link {} is already unplugged from this virtual machine", nic));
		}
	}

	if (is_on) {
		vmc->plug_link(nic);
	} else {
		vmc->unplug_link(nic);
	}
}

void VisitorInterpreterActionMachine::visit_plug_dvd(const IR::PlugDVD& plug_dvd) {
	auto path = plug_dvd.path();
	reporter.plug(vmc, "dvd", path.generic_string(), true);

	if (vmc->vm()->is_dvd_plugged()) {
		throw std::runtime_error(fmt::format("some dvd is already plugged"));
	}
	vmc->vm()->plug_dvd(path);

}

void VisitorInterpreterActionMachine::visit_unplug_dvd(const IR::PlugDVD& plug_dvd) {
	reporter.plug(vmc, "dvd", "", false);

	if (!vmc->vm()->is_dvd_plugged()) {
		std::cout << "DVD is already unplugged" << std::endl;
		// не считаем ошибкой, потому что дисковод мог быть вынут программным образом
		return;
	}
	vmc->vm()->unplug_dvd();

	auto deadline = std::chrono::steady_clock::now() +  std::chrono::seconds(10);
	while (std::chrono::steady_clock::now() < deadline) {
		if (!vmc->vm()->is_dvd_plugged()) {
			return;
		}
		timer.waitFor(std::chrono::milliseconds(300));
	}

	throw std::runtime_error(fmt::format("Timeout expired for unplugging dvd"));

}

void VisitorInterpreterActionMachine::visit_plug_flash(const IR::PlugFlash& plug_flash) {
	auto fdc = IR::program->get_flash_drive_or_throw(plug_flash.name());

	reporter.plug(vmc, "flash drive", fdc->name(), true);
	for (auto vmc: current_test->get_all_machines()) {
		if (vmc->vm()->is_flash_plugged(fdc->fd())) {
			throw std::runtime_error(fmt::format("Flash drive {} is already plugged into vm {}. You should unplug it first", fdc->name(), vmc->name()));
		}
	}

	for (auto fdc: current_test->get_all_flash_drives()) {
		if (vmc->vm()->is_flash_plugged(fdc->fd())) {
			throw std::runtime_error(fmt::format("Another flash drive {} is already plugged into vm {}. You should unplug it first", fdc->name(), vmc->name()));
		}
	}

	vmc->vm()->plug_flash_drive(fdc->fd());
}

void VisitorInterpreterActionMachine::visit_unplug_flash(const IR::PlugFlash& plug_flash) {
	auto fdc = IR::program->get_flash_drive_or_throw(plug_flash.name());

	reporter.plug(vmc, "flash drive", fdc->name(), false);
	if (!vmc->vm()->is_flash_plugged(fdc->fd())) {
		throw std::runtime_error(fmt::format("specified flash {} is already unplugged from this virtual machine", fdc->name()));
	}

	vmc->vm()->unplug_flash_drive(fdc->fd());
}

void VisitorInterpreterActionMachine::visit_plug_hostdev(const IR::PlugHostDev& plug_hostdev) {
	reporter.plug(vmc, "hostdev usb", plug_hostdev.addr(), true);
	vmc->vm()->plug_hostdev_usb(plug_hostdev.addr());
}

void VisitorInterpreterActionMachine::visit_unplug_hostdev(const IR::PlugHostDev& plug_hostdev) {
	reporter.plug(vmc, "hostdev usb", plug_hostdev.addr(), false);
	vmc->vm()->unplug_hostdev_usb(plug_hostdev.addr());
}

void VisitorInterpreterActionMachine::visit_start(const IR::Start& start) {
	try {
		reporter.start(vmc);
		vmc->vm()->start();
		auto deadline = std::chrono::steady_clock::now() +  std::chrono::milliseconds(5000);
		while (std::chrono::steady_clock::now() < deadline) {
			if (vmc->vm()->state() == VmState::Running) {
				return;
			}
			timer.waitFor(std::chrono::milliseconds(300));
		}
		throw std::runtime_error("Start timeout");
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(start.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_stop(const IR::Stop& stop) {
	try {
		reporter.stop(vmc);
		vmc->vm()->stop();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(stop.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_shutdown(const IR::Shutdown& shutdown) {
	try {
		reporter.shutdown(vmc, shutdown);
		vmc->vm()->power_button();
		auto deadline = std::chrono::steady_clock::now() +  shutdown.timeout().value();
		while (std::chrono::steady_clock::now() < deadline) {
			if (vmc->vm()->state() == VmState::Stopped) {
				return;
			}
			timer.waitFor(std::chrono::milliseconds(300));
		}
		throw std::runtime_error("Shutdown timeout");
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(shutdown.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_exec(const IR::Exec& exec) {
	try {
		reporter.exec(vmc, exec);

		if (vmc->vm()->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		auto ga = vmc->vm()->guest_additions();

		if (!ga->is_avaliable()) {
			throw std::runtime_error(fmt::format("guest additions are not installed"));
		}

		std::string script, extension, command;

		if (exec.interpreter() == "bash") {
			script = build_shell_script(exec.script());
			extension = ".sh";
			command = "bash";
		} else if (exec.interpreter() == "cmd") {
			script = build_batch_script(exec.script());
			extension = ".bat";
			command = "cmd /c";
		} else if (exec.interpreter() == "python") {
			script = build_python_script(exec.script());
			extension = ".py";
			command = "python";
		} else if (exec.interpreter() == "python2") {
			script = build_python_script(exec.script());
			extension = ".py";
			command = "python2";
		} else {
			script = build_python_script(exec.script());
			extension = ".py";
			command = "python3";
		}

		//copy the script to tmp folder
		std::hash<std::string> h;

		std::string hash = std::to_string(h(script));

		fs::path host_script_dir = fs::temp_directory_path();
		fs::path guest_script_dir = ga->get_tmp_dir();

		fs::path host_script_file = host_script_dir / std::string(hash + extension);
		fs::path guest_script_file = guest_script_dir / std::string(hash + extension);
		std::ofstream script_stream(host_script_file, std::ios::binary);
		if (!script_stream.is_open()) {
			throw std::runtime_error(fmt::format("Can't open tmp file for writing the script"));
		}

		script_stream << script;
		script_stream.close();

		ga->copy_to_guest(host_script_file, guest_script_file); //5 seconds should be enough to pass any script

		fs::remove(host_script_file.generic_string());

		command += " " + guest_script_file.generic_string();

		coro::Timeout timeout(exec.timeout().value());

		auto result = ga->execute(command, [&](const std::string& output) {
			reporter.exec_command_output(output);
		});
		if (result != 0) {
			throw std::runtime_error(exec.interpreter() + " command failed");
		}
		ga->remove_from_guest(guest_script_file);

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(exec.ast_node, current_controller));
	}
}

nlohmann::json VisitorInterpreterActionMachine::eval_js(const std::string& script, const stb::Image<stb::RGB>& screenshot) {
	try {
		auto eval_result = env->nn_client->eval_js(&screenshot, script);

		auto type = eval_result.at("type").get<std::string>();

		if (type == "error") {
			std::string message = eval_result.at("data").get<std::string>();
			throw std::runtime_error(message);
		} else if (type == "continue_error") {
			std::string message = eval_result.at("data").get<std::string>();
			throw ContinueError(message);
		} else if (type == "eval_result") {
			std::string output = eval_result.value("stdout", "");
			if (output.length()) {
				reporter.js_stdout(output);
			}
			return eval_result.at("data");
		} else {
			throw std::runtime_error(std::string("Unknown message type: ") + type);
		}
	} catch(const ContinueError& error) {
		throw;
	} catch(const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Error while executing javascript selection"));
	}
}

template <typename Func>
bool VisitorInterpreterActionMachine::screenshot_loop(Func&& func, std::chrono::milliseconds timeout, std::chrono::milliseconds interval) {
	auto deadline = std::chrono::steady_clock::now() + timeout;
	uint64_t empty_screenshots_counter = 0;

	do {
		auto start = std::chrono::high_resolution_clock::now();
		auto& screenshot = vmc->make_new_screenshot();

		if (screenshot.data) {
			empty_screenshots_counter = 0;
			bool screenshot_found = func(screenshot);
			if (screenshot_found) {
				return true;
			}
		} else {
			++empty_screenshots_counter;
			if (empty_screenshots_counter > 10) {
				throw std::runtime_error("Can't get a screenshot of the virtual machine because it's turned off");
			}
		}

		auto end = std::chrono::high_resolution_clock::now();
		if (interval > end - start) {
			timer.waitFor(interval - (end - start));
		} else {
			coro::CheckPoint();
		}
	} while (std::chrono::steady_clock::now() < deadline);

	return false;
}
