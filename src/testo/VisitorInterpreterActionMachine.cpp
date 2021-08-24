
#include <coro/CheckPoint.h>
#include <coro/Timeout.h>
#include "VisitorInterpreterActionMachine.hpp"
#include "Exceptions.hpp"
#include "IR/Program.hpp"
#include <fmt/format.h>

using namespace std::chrono_literals;

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
		fs::path from = copy.from();
		fs::path to = copy.to();

		IR::TimeInterval wait_for = copy.timeout();
		reporter.copy(current_controller, from.generic_string(), to.generic_string(), copy.ast_node->is_to_guest(), wait_for.str());

		coro::Timeout timeout(wait_for.value());

		if (vmc->vm()->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		auto ga = vmc->vm()->guest_additions();

		if (!ga->is_avaliable()) {
			throw std::runtime_error(fmt::format("guest additions are not installed"));
		}

		if(copy.ast_node->is_to_guest()) {
			//Additional check since now we can't be sure the "from" actually exists
			if (!fs::exists(from)) {
				throw std::runtime_error(std::string(copy.ast_node->begin()) + ": Error: specified path doesn't exist: " + from.generic_string());
			}
			ga->copy_to_guest(from, to);
		} else {
			ga->copy_from_guest(from, to);;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(copy.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_screenshot(const IR::Screenshot& screenshot) {
	try {
		fs::path destination = screenshot.destination();
		reporter.screenshot(vmc, destination);

		if (vmc->vm()->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		auto& screenshot = vmc->make_new_screenshot();

		if (!fs::exists(destination.parent_path())) {
			if (!fs::create_directories(destination.parent_path())) {
				throw std::runtime_error(std::string("Can't create directory: ") + destination.parent_path().generic_string());
			}
		}

		screenshot.write_png(destination.generic_string());
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(screenshot.ast_node, current_controller));
	}
}

bool VisitorInterpreterActionMachine::visit_check(const IR::Check& check) {
	try {
		IR::TimeInterval check_for = check.timeout();
		IR::TimeInterval interval = check.interval();
		auto text = template_parser.resolve(check.ast_node->select_expr->to_string(), check.stack);

		reporter.check(vmc, text, check_for.str(), interval.str());

		return screenshot_loop([&](const stb::Image<stb::RGB>& screenshot) {
			return visit_detect_expr(check.ast_node->select_expr, screenshot);
		}, check_for.value(), interval.value());

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(check.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_abort(const IR::Abort& abort) {
	if (vmc->vm()->state() == VmState::Running) {
		reporter.save_screenshot(vmc, vmc->make_new_screenshot());
	}
	throw AbortException(abort.ast_node, current_controller, abort.message());
}

void VisitorInterpreterActionMachine::visit_type(const IR::Type& type) {
	try {
		std::string text = type.text();
		if (text.size() == 0) {
			return;
		}

		IR::TimeInterval interval = type.interval();

		reporter.type(vmc, text, interval.str());

		std::vector<KeyboardCommand> commands = KeyboardManager().type(text);

		for (size_t i = 0; i < commands.size(); ++i) {
			if (i) {
				if ((commands[i-1].action == KeyboardAction::Release) &&
					(commands[i].action == KeyboardAction::Hold))
				{
					timer.waitFor(interval.value());
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
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(type.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_wait(const IR::Wait& wait) {
	try {
		IR::TimeInterval wait_for = wait.timeout();
		IR::TimeInterval interval = wait.interval();

		reporter.wait(vmc, wait.ast_node->select_expr->to_string(), wait_for.str(), interval.str());

		bool early_exit = screenshot_loop([&](const stb::Image<stb::RGB>& screenshot) {
			return visit_detect_expr(wait.ast_node->select_expr, screenshot);
		}, wait_for.value(), interval.value());

		if (!early_exit) {
			reporter.save_screenshot(vmc, vmc->get_last_screenshot());
			throw std::runtime_error("Timeout");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(wait.ast_node, current_controller));
	}
}

template <typename NNTensor>
NNTensor VisitorInterpreterActionMachine::visit_mouse_specifier_from(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const NNTensor& input) {
	auto name = specifier->name.value();
	auto arg = std::stoul(specifier->arg.value()); //should never fail since we have semantic checks

	if (name == "from_top") {
		return input.from_top(arg);
	} else if (name == "from_bottom") {
		return input.from_bottom(arg);
	} else if (name == "from_left") {
		return input.from_left(arg);
	} else if (name == "from_right") {
		return input.from_right(arg);
	}

	throw std::runtime_error("Should not be there");
}

template <typename NNTensor>
nn::Point VisitorInterpreterActionMachine::visit_mouse_specifier_centering(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const NNTensor& input) {
	auto name = specifier->name.value();

	if (name == "left_bottom") {
		return input.left_bottom();
	} else if (name == "left_center") {
		return input.left_center();
	} else if (name == "left_top") {
		return input.left_top();
	} else if (name == "center_bottom") {
		return input.center_bottom();
	} else if (name == "center") {
		return input.center();
	} else if (name == "center_top") {
		return input.center_top();
	} else if (name == "right_bottom") {
		return input.right_bottom();
	} else if (name == "right_center") {
		return input.right_center();
	} else if (name == "right_top") {
		return input.right_top();
	}

	throw std::runtime_error("Uknown center specifier");
}

template <typename NNTensor>
nn::Point VisitorInterpreterActionMachine::visit_mouse_specifier_default_centering(const NNTensor& input) {
	return input.center();
}

nn::Point VisitorInterpreterActionMachine::visit_mouse_specifier_moving(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const nn::Point& input) {
	auto name = specifier->name.value();
	auto arg = std::stoul(specifier->arg.value()); //should never fail since we have semantic checks

	if (name == "move_left") {
		return input.move_left(arg);
	} else if (name == "move_right") {
		return input.move_right(arg);
	} else if (name == "move_up") {
		return input.move_up(arg);
	} else if (name == "move_down") {
		return input.move_down(arg);
	}

	throw std::runtime_error("Should not be there");
}

template <typename NNTensor>
nn::Point VisitorInterpreterActionMachine::visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers, const NNTensor& input_) {
	size_t index = 0;

	NNTensor input = input_;

	if ((specifiers.size() > index) && specifiers[index]->is_from()) {
		input = visit_mouse_specifier_from(specifiers[index], input);
		index++;
	}

	nn::Point result;

	if (specifiers.size() > index && specifiers[index]->is_centering()) {
		result = visit_mouse_specifier_centering(specifiers[index], input);
		index++;
	} else {
		result = visit_mouse_specifier_default_centering(input);
	}

	for (size_t i = index; i < specifiers.size(); ++i) {
		result = visit_mouse_specifier_moving(specifiers[i], result);
	}

	return result;
}

nn::TextTensor VisitorInterpreterActionMachine::visit_select_text(const IR::SelectText& text, const stb::Image<stb::RGB>& screenshot) {
	auto parsed = text.text();
	return nn::find_text(&screenshot).match_text(&screenshot, parsed);
}

nn::ImgTensor VisitorInterpreterActionMachine::visit_select_img(const IR::SelectImg& img, const stb::Image<stb::RGB>& screenshot) {
	auto parsed = img.img_path();
	return nn::find_img(&screenshot, parsed);
}

nn::Homm3Tensor VisitorInterpreterActionMachine::visit_select_homm3(const IR::SelectHomm3& homm3, const stb::Image<stb::RGB>& screenshot) {
	auto parsed = homm3.id();
	return nn::find_homm3(&screenshot).match_class(&screenshot, parsed);
}

bool VisitorInterpreterActionMachine::visit_detect_js(const IR::SelectJS& js, const stb::Image<stb::RGB>& screenshot) {
	auto value = eval_js(js.script(), screenshot);

	if (value.is_bool()) {
		return (bool)value;
	} else {
	 	throw std::runtime_error("Can't process return value type. We expect a single boolean");
	}
}

nn::Point VisitorInterpreterActionMachine::visit_select_js(const IR::SelectJS& js, const stb::Image<stb::RGB>& screenshot) {
	auto value = eval_js(js.script(), screenshot);

	if (value.is_object() && !value.is_array()) {
		auto x_prop = value.get_property_str("x");
		if (x_prop.is_undefined()) {
			throw std::runtime_error("Object doesn't have the x propery");
		}

		auto y_prop = value.get_property_str("y");
		if (y_prop.is_undefined()) {
			throw std::runtime_error("Object doesn't have the y propery");
		}

		nn::Point point;
		point.x = x_prop;
		point.y = y_prop;
		return point;
	} else {
		throw std::runtime_error("Can't process return value type. We expect a single object");
	}
}

bool VisitorInterpreterActionMachine::VisitorInterpreterActionMachine::visit_detect_expr(std::shared_ptr<AST::SelectExpr> select_expr, const stb::Image<stb::RGB>& screenshot)  {
	if (auto p = std::dynamic_pointer_cast<AST::SelectNegationExpr>(select_expr)) {
		return !visit_detect_expr(p->expr, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectText>(select_expr)) {
		return (bool)visit_select_text({p, stack}, screenshot).size();
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectJS>(select_expr)) {
		return visit_detect_js({p, stack}, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectImg>(select_expr)) {
		return (bool)visit_select_img({p, stack}, screenshot).size();
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectHomm3>(select_expr)) {
		return (bool)visit_select_homm3({p, stack}, screenshot).size();
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectParentedExpr>(select_expr)) {
		return visit_detect_expr(p->select_expr, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectBinOp>(select_expr)) {
		return visit_detect_binop(p, screenshot);
	} else {
		throw std::runtime_error("Unknown select expression type");
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

		for (auto key_spec: press.ast_node->keys) {
			visit_key_spec({key_spec, stack}, interval.value());
			timer.waitFor(interval.value());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(press.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_hold(const IR::Hold& hold) {
	try {
		auto buttons = hold.buttons();

		reporter.hold_key(vmc, hold.ast_node->combination->to_string());
		for (KeyboardButton button: buttons) {
			vmc->hold(button);
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(hold.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_release(const IR::Release& release) {
	try {
		auto buttons = release.buttons();

		if (buttons.size()) {
			reporter.release_key(vmc, release.ast_node->combination->to_string());
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
	IR::TimeInterval timeout = mouse_selectable.timeout();
	std::string where_to_go = mouse_selectable.where_to_go();

	for (auto specifier: mouse_selectable.ast_node->mouse_additional_specifiers) {
		where_to_go += specifier->to_string();
	}

	reporter.mouse_move_click_selectable(vmc, where_to_go, timeout.str());

	bool early_exit = screenshot_loop([&](const stb::Image<stb::RGB>& screenshot) {
		try {
			nn::Point point;
			if (auto p = std::dynamic_pointer_cast<AST::SelectJS>(mouse_selectable.ast_node->basic_select_expr)) {
				point = visit_select_js({p, stack}, screenshot);
			} else if (auto p = std::dynamic_pointer_cast<AST::SelectText>(mouse_selectable.ast_node->basic_select_expr)) {
				auto tensor = visit_select_text({p, stack}, screenshot);
				//each specifier can throw an exception if something goes wrong.
				point = visit_mouse_additional_specifiers(mouse_selectable.ast_node->mouse_additional_specifiers, tensor);
			} else if (auto p = std::dynamic_pointer_cast<AST::SelectImg>(mouse_selectable.ast_node->basic_select_expr)) {
				auto tensor = visit_select_img({p, stack}, screenshot);
				//each specifier can throw an exception if something goes wrong.
				point = visit_mouse_additional_specifiers(mouse_selectable.ast_node->mouse_additional_specifiers, tensor);
			} else if (auto p = std::dynamic_pointer_cast<AST::SelectHomm3>(mouse_selectable.ast_node->basic_select_expr)) {
				auto tensor = visit_select_homm3({p, stack}, screenshot);
				//each specifier can throw an exception if something goes wrong.
				point = visit_mouse_additional_specifiers(mouse_selectable.ast_node->mouse_additional_specifiers, tensor);
			}
			vmc->vm()->mouse_move_abs(point.x, point.y);
			return true;
		} catch (const nn::LogicError&) {
			reporter.save_screenshot(vmc, screenshot);
			throw;
		} catch (const nn::ContinueError&) {
			return false;
		}
	}, timeout.value(), 1s);

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
		reporter.mouse_move_click(vmc, mouse_move_click.event_type());

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
	auto x = coordinates.x();
	auto y = coordinates.y();
	reporter.mouse_move_click_coordinates(vmc, x, y);
	if (coordinates.x_is_relative() && coordinates.y_is_relative()) {
		vmc->vm()->mouse_move_rel(std::stoi(x), std::stoi(y));
	} else if (!coordinates.x_is_relative() && !coordinates.y_is_relative()) {
		vmc->vm()->mouse_move_abs(std::stoul(x), std::stoul(y));
	} else {
		throw std::runtime_error("Should not be there");
	}
}

void VisitorInterpreterActionMachine::visit_mouse_hold(const IR::MouseHold& mouse_hold) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.mouse_hold(vmc, mouse_hold.ast_node->button.value());
		vmc->mouse_hold(mouse_hold.button());
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_hold.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_mouse_release(const IR::MouseRelease& mouse_release) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.mouse_release(vmc);
		vmc->mouse_release();
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_release.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_mouse_wheel(std::shared_ptr<AST::MouseWheel> mouse_wheel) {
	try {
		reporter.mouse_wheel(vmc, mouse_wheel->direction.value());

		auto mouse_press = [&](MouseButton button) {
			vmc->mouse_hold(button);
			timer.waitFor(std::chrono::milliseconds(60));
			vmc->mouse_release();
		};

		if (mouse_wheel->direction.value() == "up") {
			mouse_press(MouseButton::WheelUp);
		} else if (mouse_wheel->direction.value() == "down") {
			mouse_press(MouseButton::WheelDown);
		} else {
			throw std::runtime_error("Unknown wheel direction");
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_wheel, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_key_spec(const IR::KeySpec& key_spec, std::chrono::milliseconds interval) {
	std::vector<KeyboardButton> buttons = key_spec.buttons();
	uint32_t times = key_spec.times();

	reporter.press_key(vmc, key_spec.ast_node->combination->to_string(), times);

	for (uint32_t i = 0; i < times; i++) {
		for (auto it = buttons.begin(); it != buttons.end(); ++it) {
			vmc->hold(*it);
		}
		for (auto it = buttons.rbegin(); it != buttons.rend(); ++it) {
			vmc->release(*it);
		}
		timer.waitFor(interval);
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
			throw std::runtime_error(std::string("unknown hardware to plug/unplug: ") +
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
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		IR::TimeInterval wait_for = shutdown.timeout();
		reporter.shutdown(vmc, wait_for.str());
		vmc->vm()->power_button();
		auto deadline = std::chrono::steady_clock::now() +  wait_for.value();
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
		reporter.exec(vmc, exec.interpreter(), exec.timeout().str());

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

js::Value VisitorInterpreterActionMachine::eval_js(const std::string& script, const stb::Image<stb::RGB>& screenshot) {
	try {
		js_current_ctx.reset(new js::Context(&screenshot));
		return js_current_ctx->eval(script);
	} catch (const nn::ContinueError& error) {
		throw;
	}
	catch(const std::exception& error) {
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
