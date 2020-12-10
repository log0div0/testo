
#include "VisitorInterpreterActionMachine.hpp"
#include "Exceptions.hpp"
#include "IR/Program.hpp"
#include "coro/CheckPoint.h"
#include "coro/Timeout.h"
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
	charmap.insert({
		{U'0', {"ZERO"}},
		{U'1', {"ONE"}},
		{U'2', {"TWO"}},
		{U'3', {"THREE"}},
		{U'4', {"FOUR"}},
		{U'5', {"FIVE"}},
		{U'6', {"SIX"}},
		{U'7', {"SEVEN"}},
		{U'8', {"EIGHT"}},
		{U'9', {"NINE"}},
		{U')', {"LEFTSHIFT", "ZERO"}},
		{U'!', {"LEFTSHIFT", "ONE"}},
		{U'@', {"LEFTSHIFT", "TWO"}},
		{U'#', {"LEFTSHIFT", "THREE"}},
		{U'$', {"LEFTSHIFT", "FOUR"}},
		{U'%', {"LEFTSHIFT", "FIVE"}},
		{U'^', {"LEFTSHIFT", "SIX"}},
		{U'&', {"LEFTSHIFT", "SEVEN"}},
		{U'*', {"LEFTSHIFT", "EIGHT"}},
		{U'(', {"LEFTSHIFT", "NINE"}},
		{U'a', {"A"}},
		{U'b', {"B"}},
		{U'c', {"C"}},
		{U'd', {"D"}},
		{U'e', {"E"}},
		{U'f', {"F"}},
		{U'g', {"G"}},
		{U'h', {"H"}},
		{U'i', {"I"}},
		{U'j', {"J"}},
		{U'k', {"K"}},
		{U'l', {"L"}},
		{U'm', {"M"}},
		{U'n', {"N"}},
		{U'o', {"O"}},
		{U'p', {"P"}},
		{U'q', {"Q"}},
		{U'r', {"R"}},
		{U's', {"S"}},
		{U't', {"T"}},
		{U'u', {"U"}},
		{U'v', {"V"}},
		{U'w', {"W"}},
		{U'x', {"X"}},
		{U'y', {"Y"}},
		{U'z', {"Z"}},
		{U'A', {"LEFTSHIFT", "A"}},
		{U'B', {"LEFTSHIFT", "B"}},
		{U'C', {"LEFTSHIFT", "C"}},
		{U'D', {"LEFTSHIFT", "D"}},
		{U'E', {"LEFTSHIFT", "E"}},
		{U'F', {"LEFTSHIFT", "F"}},
		{U'G', {"LEFTSHIFT", "G"}},
		{U'H', {"LEFTSHIFT", "H"}},
		{U'I', {"LEFTSHIFT", "I"}},
		{U'J', {"LEFTSHIFT", "J"}},
		{U'K', {"LEFTSHIFT", "K"}},
		{U'L', {"LEFTSHIFT", "L"}},
		{U'M', {"LEFTSHIFT", "M"}},
		{U'N', {"LEFTSHIFT", "N"}},
		{U'O', {"LEFTSHIFT", "O"}},
		{U'P', {"LEFTSHIFT", "P"}},
		{U'Q', {"LEFTSHIFT", "Q"}},
		{U'R', {"LEFTSHIFT", "R"}},
		{U'S', {"LEFTSHIFT", "S"}},
		{U'T', {"LEFTSHIFT", "T"}},
		{U'U', {"LEFTSHIFT", "U"}},
		{U'V', {"LEFTSHIFT", "V"}},
		{U'W', {"LEFTSHIFT", "W"}},
		{U'X', {"LEFTSHIFT", "X"}},
		{U'Y', {"LEFTSHIFT", "Y"}},
		{U'Z', {"LEFTSHIFT", "Z"}},

		{U'а', {"F"}},
		{U'б', {"COMMA"}},
		{U'в', {"D"}},
		{U'г', {"U"}},
		{U'д', {"L"}},
		{U'е', {"T"}},
		{U'ё', {"GRAVE"}},
		{U'ж', {"SEMICOLON"}},
		{U'з', {"P"}},
		{U'и', {"B"}},
		{U'й', {"Q"}},
		{U'к', {"R"}},
		{U'л', {"K"}},
		{U'м', {"V"}},
		{U'н', {"Y"}},
		{U'о', {"J"}},
		{U'п', {"G"}},
		{U'р', {"H"}},
		{U'с', {"C"}},
		{U'т', {"N"}},
		{U'у', {"E"}},
		{U'ф', {"A"}},
		{U'х', {"LEFTBRACE"}},
		{U'ц', {"W"}},
		{U'ч', {"X"}},
		{U'ш', {"I"}},
		{U'щ', {"O"}},
		{U'ъ', {"RIGHTBRACE"}},
		{U'ы', {"S"}},
		{U'ь', {"M"}},
		{U'э', {"APOSTROPHE"}},
		{U'ю', {"DOT"}},
		{U'я', {"Z"}},

		{U'А', {"LEFTSHIFT", "F"}},
		{U'Б', {"LEFTSHIFT", "COMMA"}},
		{U'В', {"LEFTSHIFT", "D"}},
		{U'Г', {"LEFTSHIFT", "U"}},
		{U'Д', {"LEFTSHIFT", "L"}},
		{U'Е', {"LEFTSHIFT", "T"}},
		{U'Ё', {"LEFTSHIFT", "GRAVE"}},
		{U'Ж', {"LEFTSHIFT", "SEMICOLON"}},
		{U'З', {"LEFTSHIFT", "P"}},
		{U'И', {"LEFTSHIFT", "B"}},
		{U'Й', {"LEFTSHIFT", "Q"}},
		{U'К', {"LEFTSHIFT", "R"}},
		{U'Л', {"LEFTSHIFT", "K"}},
		{U'М', {"LEFTSHIFT", "V"}},
		{U'Н', {"LEFTSHIFT", "Y"}},
		{U'О', {"LEFTSHIFT", "J"}},
		{U'П', {"LEFTSHIFT", "G"}},
		{U'Р', {"LEFTSHIFT", "H"}},
		{U'С', {"LEFTSHIFT", "C"}},
		{U'Т', {"LEFTSHIFT", "N"}},
		{U'У', {"LEFTSHIFT", "E"}},
		{U'Ф', {"LEFTSHIFT", "A"}},
		{U'Х', {"LEFTSHIFT", "LEFTBRACE"}},
		{U'Ц', {"LEFTSHIFT", "W"}},
		{U'Ч', {"LEFTSHIFT", "X"}},
		{U'Ш', {"LEFTSHIFT", "I"}},
		{U'Щ', {"LEFTSHIFT", "O"}},
		{U'Ъ', {"LEFTSHIFT", "RIGHTBRACE"}},
		{U'Ы', {"LEFTSHIFT", "S"}},
		{U'Ь', {"LEFTSHIFT", "M"}},
		{U'Э', {"LEFTSHIFT", "APOSTROPHE"}},
		{U'Ю', {"LEFTSHIFT", "DOT"}},
		{U'Я', {"LEFTSHIFT", "Z"}},

		{U'-', {"MINUS"}},
		{U'_', {"LEFTSHIFT", "MINUS"}},
		{U'=', {"EQUALSIGN"}},
		{U'+', {"LEFTSHIFT", "EQUALSIGN"}},
		{U'\'', {"APOSTROPHE"}},
		{U'\"', {"LEFTSHIFT", "APOSTROPHE"}},
		{U'\\', {"BACKSLASH"}},
		{U'\n', {"ENTER"}},
		{U'\t', {"TAB"}},
		{U'|', {"LEFTSHIFT", "BACKSLASH"}},
		{U',', {"COMMA"}},
		{U'<', {"LEFTSHIFT", "COMMA"}},
		{U'.', {"DOT"}},
		{U'>', {"LEFTSHIFT", "DOT"}},
		{U'/', {"SLASH"}},
		{U'?', {"LEFTSHIFT", "SLASH"}},
		{U';', {"SEMICOLON"}},
		{U':', {"LEFTSHIFT", "SEMICOLON"}},
		{U'[', {"LEFTBRACE"}},
		{U'{', {"LEFTSHIFT", "LEFTBRACE"}},
		{U']', {"RIGHTBRACE"}},
		{U'}', {"LEFTSHIFT", "RIGHTBRACE"}},
		{U'`', {"GRAVE"}},
		{U'~', {"LEFTSHIFT", "GRAVE"}},
		{U' ', {"SPACE"}}
	});
}

void VisitorInterpreterActionMachine::visit_action(std::shared_ptr<AST::IAction> action) {
	if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Abort>>(action)) {
		visit_abort({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Print>>(action)) {
		visit_print({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Type>>(action)) {
		visit_type({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Wait>>(action)) {
		visit_wait({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Sleep>>(action)) {
		visit_sleep({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Press>>(action)) {
		visit_press({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Hold>>(action)) {
		visit_hold({p->action, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Release>>(action)) {
		visit_release({p->action, stack});
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
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::MacroCall>>(action)) {
		visit_macro_call(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::IfClause>>(action)) {
		visit_if_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ForClause>>(action)) {
		visit_for_clause(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::CycleControl>>(action)) {
		throw CycleControlException(p->action->t);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::ActionBlock>>(action)) {
		visit_action_block(p->action);
	} else if (auto p = std::dynamic_pointer_cast<AST::Action<AST::Empty>>(action)) {
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

		std::string wait_for = copy.timeout();
		reporter.copy(current_controller, from.generic_string(), to.generic_string(), copy.ast_node->is_to_guest(), wait_for);

		coro::Timeout timeout(std::chrono::milliseconds(time_to_milliseconds(wait_for)));

		if (vmc->vm()->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		if (!vmc->vm()->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions are not installed"));
		}

		if(copy.ast_node->is_to_guest()) {
			vmc->vm()->copy_to_guest(from, to);
		} else {
			vmc->vm()->copy_from_guest(from, to);;
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(copy.ast_node, current_controller));
	}
}

bool VisitorInterpreterActionMachine::visit_check(const IR::Check& check) {
	try {
		std::string check_for = check.timeout();
		std::string interval_str = check.interval();
		auto interval = std::chrono::milliseconds(time_to_milliseconds(interval_str));
		auto text = template_parser.resolve(std::string(*check.ast_node->select_expr), check.stack);

		reporter.check(vmc, text, check_for, interval_str);

		auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_to_milliseconds(check_for));

		do {
			auto start = std::chrono::high_resolution_clock::now();
			auto screenshot = vmc->vm()->screenshot();

			if (visit_detect_expr(check.ast_node->select_expr, screenshot)) {
				return true;
			}

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time = end - start;
			//std::cout << "time = " << time.count() << " seconds" << std::endl;
			if (interval > end - start) {
				timer.waitFor(interval - (end - start));
			} else {
				coro::CheckPoint();
			}
		} while (std::chrono::steady_clock::now() < deadline);

		return false;
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(check.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_abort(const IR::Abort& abort) {
	reporter.save_screenshot(vmc, vmc->vm()->screenshot());
	throw AbortException(abort.ast_node, current_controller, abort.message());
}

static std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;

void VisitorInterpreterActionMachine::visit_type(const IR::Type& type) {
	try {
		std::string text = type.text();
		if (text.size() == 0) {
			return;
		}

		std::string interval = type.interval();

		reporter.type(vmc, text, interval);

		for (char32_t c: conv.from_bytes(text)) {
			auto buttons = charmap.find(c);
			if (buttons == charmap.end()) {
				throw std::runtime_error("Unknown character to type");
			}
			vmc->press(buttons->second);
			timer.waitFor(std::chrono::milliseconds(time_to_milliseconds(interval)));
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(type.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_wait(const IR::Wait& wait) {
	try {
		std::string wait_for = wait.timeout();
		std::string interval_str = wait.interval();
		auto interval = std::chrono::milliseconds(time_to_milliseconds(interval_str));
		auto text = wait.select_expr();

		reporter.wait(vmc, text, wait_for, interval_str);

		auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_to_milliseconds(wait_for));
		stb::Image<stb::RGB> screenshot;

		do {
			auto start = std::chrono::high_resolution_clock::now();
			screenshot = vmc->vm()->screenshot();

			if (visit_detect_expr(wait.ast_node->select_expr, screenshot)) {
				return;
			}

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time = end - start;
			//std::cout << "time = " << time.count() << " seconds" << std::endl;
			if (interval > end - start) {
				timer.waitFor(interval - (end - start));
			} else {
				coro::CheckPoint();
			}
		} while (std::chrono::steady_clock::now() < deadline);

		reporter.save_screenshot(vmc, screenshot);
		throw std::runtime_error("Timeout");

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

nn::TextTensor VisitorInterpreterActionMachine::visit_select_text(const IR::SelectText& text, stb::Image<stb::RGB>& screenshot) {
	auto parsed = text.text();
	return nn::find_text(&screenshot).match_text(&screenshot, parsed);
}

nn::ImgTensor VisitorInterpreterActionMachine::visit_select_img(const IR::SelectImg& img, stb::Image<stb::RGB>& screenshot) {
	auto parsed = img.img_path();
	return nn::find_img(&screenshot, parsed);
}

bool VisitorInterpreterActionMachine::visit_detect_js(const IR::SelectJS& js, stb::Image<stb::RGB>& screenshot) {
	auto value = eval_js(js.script(), screenshot);

	if (value.is_bool()) {
		return (bool)value;
	} else {
	 	throw std::runtime_error("Can't process return value type. We expect a single boolean");
	}
}

nn::Point VisitorInterpreterActionMachine::visit_select_js(const IR::SelectJS& js, stb::Image<stb::RGB>& screenshot) {
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

bool VisitorInterpreterActionMachine::VisitorInterpreterActionMachine::visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr, stb::Image<stb::RGB>& screenshot)  {
	if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::ISelectable>>(select_expr)) {
		return visit_detect_selectable(p->select_expr, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectExpr<AST::SelectBinOp>>(select_expr)) {
		return visit_detect_binop(p->select_expr, screenshot);
	} else {
		throw std::runtime_error("Unknown select expression type");
	}
}


bool VisitorInterpreterActionMachine::visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable, stb::Image<stb::RGB>& screenshot) {
	bool is_negated = selectable->is_negated();

	if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectText>>(selectable)) {
		return is_negated ^ (bool)visit_select_text({p->selectable, stack}, screenshot).size();
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(selectable)) {
		return is_negated ^ visit_detect_js({p->selectable, stack}, screenshot);
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectImg>>(selectable)) {
		return is_negated ^ (bool)visit_select_img({p->selectable, stack}, screenshot).size();
	} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectParentedExpr>>(selectable)) {
		return is_negated ^ visit_detect_expr(p->selectable->select_expr, screenshot);
	}  else {
		throw std::runtime_error("Unknown selectable type");
	}
}

bool VisitorInterpreterActionMachine::visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop, stb::Image<stb::RGB>& screenshot) {
	auto left_value = visit_detect_expr(binop->left, screenshot);
	if (binop->t.type() == Token::category::double_ampersand) {
		if (!left_value) {
			return false;
		} else {
			return left_value && visit_detect_expr(binop->right, screenshot);
		}
	} else if (binop->t.type() == Token::category::double_vertical_bar) {
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
		std::string interval = press.interval();
		auto press_interval = time_to_milliseconds(interval);

		for (auto key_spec: press.ast_node->keys) {
			visit_key_spec(key_spec, press_interval);
			timer.waitFor(std::chrono::milliseconds(press_interval));
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(press.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_hold(const IR::Hold& hold) {
	try {
		reporter.hold_key(vmc, std::string(*hold.ast_node->combination));
		vmc->hold(hold.buttons());
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(hold.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_release(const IR::Release& release) {
	try {
		auto buttons = release.buttons();

		if (buttons.size()) {
			reporter.release_key(vmc, std::string(*release.ast_node->combination));
			vmc->release(release.buttons());
		} else {
			reporter.release_key(vmc);
			vmc->release();
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(release.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable) {
	std::string timeout = mouse_selectable.timeout();
	std::string where_to_go = mouse_selectable.where_to_go();

	for (auto specifier: mouse_selectable.ast_node->specifiers) {
		where_to_go += std::string(*specifier);
	}

	reporter.mouse_move_click_selectable(vmc, where_to_go, timeout);

	auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_to_milliseconds(timeout));
	stb::Image<stb::RGB> screenshot;

	do {
		auto start = std::chrono::high_resolution_clock::now();
		screenshot = vmc->vm()->screenshot();
		try {
			nn::Point point;
			if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectJS>>(mouse_selectable.ast_node->selectable)) {
				point = visit_select_js({p->selectable, stack}, screenshot);
			} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectText>>(mouse_selectable.ast_node->selectable)) {
				auto tensor = visit_select_text({p->selectable, stack}, screenshot);
				//each specifier can throw an exception if something goes wrong.
				point = visit_mouse_additional_specifiers(mouse_selectable.ast_node->specifiers, tensor);
			} else if (auto p = std::dynamic_pointer_cast<AST::Selectable<AST::SelectImg>>(mouse_selectable.ast_node->selectable)) {
				auto tensor = visit_select_img({p->selectable, stack}, screenshot);
				//each specifier can throw an exception if something goes wrong.
				point = visit_mouse_additional_specifiers(mouse_selectable.ast_node->specifiers, tensor);
			}
			vmc->vm()->mouse_move_abs(point.x, point.y);
			return;
		} catch (const nn::LogicError&) {
			reporter.save_screenshot(vmc, screenshot);
			throw;
		} catch (const nn::ContinueError&) {
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> time = end - start;
			if (time < 1s) {
				timer.waitFor(std::chrono::duration_cast<std::chrono::milliseconds>(1s - time));
			} else {
				coro::CheckPoint();
			}
			continue;
		}
	} while (std::chrono::steady_clock::now() < deadline);

	reporter.save_screenshot(vmc, screenshot);
	throw std::runtime_error("Timeout");
}

void VisitorInterpreterActionMachine::visit_mouse(const IR::Mouse& mouse) {
	if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseMoveClick>>(mouse.ast_node->event)) {
		return visit_mouse_move_click({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseHold>>(mouse.ast_node->event)) {
		return visit_mouse_hold({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseRelease>>(mouse.ast_node->event)) {
		return visit_mouse_release({p->event, stack});
	} else if (auto p = std::dynamic_pointer_cast<AST::MouseEvent<AST::MouseWheel>>(mouse.ast_node->event)) {
		throw std::runtime_error("Not implemented yet");
	} else {
		throw std::runtime_error("Unknown mouse actions");
	}
}

void VisitorInterpreterActionMachine::visit_mouse_move_click(const IR::MouseMoveClick& mouse_move_click) {
	try {
		reporter.mouse_move_click(vmc, mouse_move_click.event_type());

		if (mouse_move_click.ast_node->object) {
			if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseCoordinates>>(mouse_move_click.ast_node->object)) {
				visit_mouse_move_coordinates({p->target, stack});
			} else if (auto p = std::dynamic_pointer_cast<AST::MouseMoveTarget<AST::MouseSelectable>>(mouse_move_click.ast_node->object)) {
				visit_mouse_move_selectable({p->target, stack});
			} else {
				throw std::runtime_error("Unknown mouse move target");
			}
		} else {
			reporter.mouse_no_object();
		}

		if (mouse_move_click.event_type() == "move") {
			return;
		}

		if (mouse_move_click.event_type() == "click" || mouse_move_click.event_type() == "lclick") {
			vmc->mouse_press({MouseButton::Left});
		} else if (mouse_move_click.event_type() == "rclick") {
			vmc->mouse_press({MouseButton::Right});
		} else if (mouse_move_click.event_type() == "mclick") {
			vmc->mouse_press({MouseButton::Middle});
		} else if (mouse_move_click.event_type() == "dclick") {
			vmc->mouse_press({MouseButton::Left});
			timer.waitFor(std::chrono::milliseconds(60));
			vmc->mouse_press({MouseButton::Left});
		} else {
			throw std::runtime_error("Unsupported click type");
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_move_click.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates) {
	auto dx = coordinates.x();
	auto dy = coordinates.y();
	reporter.mouse_move_click_coordinates(vmc, dx, dy);
	if ((dx[0] == '+') || (dx[0] == '-')) {
		vmc->vm()->mouse_move_rel("x", std::stoi(dx));
	} else {
		vmc->vm()->mouse_move_abs("x", std::stoul(dx));
	}

	if ((dy[0] == '+') || (dy[0] == '-')) {
		vmc->vm()->mouse_move_rel("y", std::stoi(dy));
	} else {
		vmc->vm()->mouse_move_abs("y", std::stoul(dy));
	}
}

void VisitorInterpreterActionMachine::visit_mouse_hold(const IR::MouseHold& mouse_hold) {
	try {
		auto vmc = std::dynamic_pointer_cast<IR::Machine>(current_controller);
		reporter.mouse_hold(vmc, mouse_hold.button());
		if (mouse_hold.button() == "lbtn") {
			vmc->mouse_hold({MouseButton::Left});
		} else if (mouse_hold.button() == "rbtn") {
			vmc->mouse_hold({MouseButton::Right});
		} else if (mouse_hold.button() == "mbtn") {
			vmc->mouse_hold({MouseButton::Middle});
		} else {
			throw std::runtime_error("Unknown mouse button: " + mouse_hold.button());
		}
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

		if (mouse_wheel->direction.value() == "up") {
			vmc->mouse_press({MouseButton::WheelUp});
		} else if (mouse_wheel->direction.value() == "down") {
			vmc->mouse_press({MouseButton::WheelDown});
		} else {
			throw std::runtime_error("Unknown wheel direction");
		}

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(mouse_wheel, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec, uint32_t interval) {
	uint32_t times = key_spec->get_times();

	reporter.press_key(vmc, *key_spec->combination, times);

	for (uint32_t i = 0; i < times; i++) {
		vmc->press(key_spec->combination->get_buttons());
		timer.waitFor(std::chrono::milliseconds(interval));
	}
}

void VisitorInterpreterActionMachine::visit_plug(const IR::Plug& plug) {
	try {
		if (plug.entity_type() == "nic") {
			return visit_plug_nic(plug);
		} else if (plug.entity_type() == "link") {
			return visit_plug_link(plug);
		} else if (plug.entity_type() == "dvd") {
			return visit_plug_dvd(plug);
		} else if (plug.entity_type() == "flash") {
			if(plug.is_on()) {
				return visit_plug_flash(plug);
			} else {
				return visit_unplug_flash(plug);
			}
		} else {
			throw std::runtime_error(std::string("unknown hardware type to plug/unplug: ") +
				plug.entity_type());
		}
	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(plug.ast_node, current_controller));
	}
}

void VisitorInterpreterActionMachine::visit_plug_nic(const IR::Plug& plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vmc while semantic analisys
	auto nic = plug.entity_name();

	reporter.plug(vmc, "nic", nic, plug.is_on());

	auto nics = vmc->vm()->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("specified nic {} is not present in this virtual machine", nic));
	}

	if (vmc->vm()->state() != VmState::Stopped) {
		throw std::runtime_error(fmt::format("virtual machine is running, but must be stopped"));
	}

	if (vmc->is_nic_plugged(nic) == plug.is_on()) {
		if (plug.is_on()) {
			throw std::runtime_error(fmt::format("specified nic {} is already plugged in this virtual machine", nic));
		} else {
			throw std::runtime_error(fmt::format("specified nic {} is not unplugged from this virtual machine", nic));
		}
	}

	if (plug.is_on()) {
		vmc->plug_nic(nic);
	} else {
		vmc->unplug_nic(nic);
	}
}

void VisitorInterpreterActionMachine::visit_plug_link(const IR::Plug& plug) {
	//we have to do it only while interpreting because we can't be sure we know
	//the vmc while semantic analisys

	auto nic = plug.entity_name();

	reporter.plug(vmc, "link", nic, plug.is_on());

	auto nics = vmc->vm()->nics();
	if (nics.find(nic) == nics.end()) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is not present in this virtual machine", nic));
	}

	if (!vmc->is_nic_plugged(nic)) {
		throw std::runtime_error(fmt::format("the nic for specified link {} is unplugged, you must to plug it first", nic));
	}

	if (plug.is_on() == vmc->is_link_plugged(nic)) {
		if (plug.is_on()) {
			throw std::runtime_error(fmt::format("specified link {} is already plugged in this virtual machine", nic));
		} else {
			throw std::runtime_error(fmt::format("specified link {} is already unplugged from this virtual machine", nic));
		}
	}

	if (plug.is_on()) {
		vmc->plug_link(nic);
	} else {
		vmc->unplug_link(nic);
	}
}

void VisitorInterpreterActionMachine::visit_plug_dvd(const IR::Plug& plug) {
	if (plug.is_on()) {
		auto path = plug.dvd_path();
		reporter.plug(vmc, "dvd", path.generic_string(), true);

		if (vmc->vm()->is_dvd_plugged()) {
			throw std::runtime_error(fmt::format("some dvd is already plugged"));
		}
		vmc->vm()->plug_dvd(path);
	} else {
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
}

void VisitorInterpreterActionMachine::visit_plug_flash(const IR::Plug& plug) {
	auto fdc = IR::program->get_flash_drive_or_throw(plug.entity_name());

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

void VisitorInterpreterActionMachine::visit_unplug_flash(const IR::Plug& plug) {
	auto fdc = IR::program->get_flash_drive_or_throw(plug.entity_name());

	reporter.plug(vmc, "flash drive", fdc->name(), false);
	if (!vmc->vm()->is_flash_plugged(fdc->fd())) {
		throw std::runtime_error(fmt::format("specified flash {} is already unplugged from this virtual machine", fdc->name()));
	}

	vmc->vm()->unplug_flash_drive(fdc->fd());
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
		std::string wait_for = shutdown.timeout();
		reporter.shutdown(vmc, wait_for);
		vmc->vm()->power_button();
		auto deadline = std::chrono::steady_clock::now() +  std::chrono::milliseconds(time_to_milliseconds(wait_for));
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
		reporter.exec(vmc, exec.interpreter(), exec.timeout());

		if (vmc->vm()->state() != VmState::Running) {
			throw std::runtime_error(fmt::format("virtual machine is not running"));
		}

		if (!vmc->vm()->is_additions_installed()) {
			throw std::runtime_error(fmt::format("guest additions is not installed"));
		}

		std::string script, extension, interpreter;
		std::vector<std::string> args;

		if (exec.interpreter() == "bash") {
			script = build_shell_script(exec.script());
			extension = ".sh";
			interpreter = "bash";
		} else if (exec.interpreter() == "cmd") {
			script = build_batch_script(exec.script());
			extension = ".bat";
			interpreter = "cmd";
			args.push_back("/c");
		} else if (exec.interpreter() == "python") {
			script = build_python_script(exec.script());
			extension = ".py";
			interpreter = "python";
		} else if (exec.interpreter() == "python2") {
			script = build_python_script(exec.script());
			extension = ".py";
			interpreter = "python2";
		} else {
			script = build_python_script(exec.script());
			extension = ".py";
			interpreter = "python3";
		}

		//copy the script to tmp folder
		std::hash<std::string> h;

		std::string hash = std::to_string(h(script));

		fs::path host_script_dir = fs::temp_directory_path();
		fs::path guest_script_dir = vmc->vm()->get_tmp_dir();

		fs::path host_script_file = host_script_dir / std::string(hash + extension);
		fs::path guest_script_file = guest_script_dir / std::string(hash + extension);
		std::ofstream script_stream(host_script_file, std::ios::binary);
		if (!script_stream.is_open()) {
			throw std::runtime_error(fmt::format("Can't open tmp file for writing the script"));
		}

		script_stream << script;
		script_stream.close();

		vmc->vm()->copy_to_guest(host_script_file, guest_script_file); //5 seconds should be enough to pass any script

		fs::remove(host_script_file.generic_string());

		args.push_back(guest_script_file.generic_string());

		coro::Timeout timeout(std::chrono::milliseconds(time_to_milliseconds(exec.timeout())));

		auto result = vmc->vm()->run(interpreter, args, [&](const std::string& output) {
			reporter.exec_command_output(output);
		});
		if (result != 0) {
			throw std::runtime_error(interpreter + " command failed");
		}
		vmc->vm()->remove_from_guest(guest_script_file);

	} catch (const std::exception& error) {
		std::throw_with_nested(ActionException(exec.ast_node, current_controller));
	}
}

js::Value VisitorInterpreterActionMachine::eval_js(const std::string& script, stb::Image<stb::RGB>& screenshot) {
	try {
		js_current_ctx.reset(new js::Context(&screenshot));
		return js_current_ctx->eval(script);
	} catch (const nn::ContinueError& error) {
		throw error;
	}
	catch(const std::exception& error) {
		std::throw_with_nested(std::runtime_error("Error while executing javascript selection"));
	}
}