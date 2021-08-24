
#include "Action.hpp"
#include "Program.hpp"
#include "../TemplateLiterals.hpp"
#include "../Exceptions.hpp"

namespace IR {

std::string Abort::message() const {
	return String(ast_node->message, stack).text();
}

std::string Print::message() const {
	return String(ast_node->message, stack).text();
}

TimeInterval Press::interval() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("interval", "TESTO_PRESS_DEFAULT_INTERVAL");
}

std::vector<KeyboardButton> KeySpec::buttons() const {
	std::vector<KeyboardButton> result;
	for (auto& button: ast_node->combination->get_buttons()) {
		result.push_back(ToKeyboardButton(button));
	}
	return result;
}

int32_t KeySpec::times() const {
	if (ast_node->times) {
		return Number(ast_node->times, stack).value();
	} else {
		return 1;
	}
}

std::vector<KeyboardButton> Hold::buttons() const {
	std::vector<KeyboardButton> result;
	for (auto& button: ast_node->combination->get_buttons()) {
		result.push_back(ToKeyboardButton(button));
	}
	return result;
}

std::vector<KeyboardButton> Release::buttons() const {
	if (ast_node->combination) {
		std::vector<KeyboardButton> result;
		for (auto& button: ast_node->combination->get_buttons()) {
			result.push_back(ToKeyboardButton(button));
		}
		return result;
	}
	return {};
}

std::string Type::text() const {
	return String(ast_node->text, stack).text();
}

TimeInterval Type::interval() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("interval", "TESTO_TYPE_DEFAULT_INTERVAL");
}

TimeInterval Wait::timeout() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("timeout", "TESTO_WAIT_DEFAULT_TIMEOUT");
}

TimeInterval Wait::interval() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("interval", "TESTO_WAIT_DEFAULT_INTERVAL");
}

TimeInterval Sleep::timeout() const {
	return {ast_node->timeout, stack};
}

std::string MouseMoveClick::event_type() const {
	return ast_node->event.value();
}

std::string MouseCoordinates::x() const {
	return ast_node->dx.value();
}

std::string MouseCoordinates::y() const {
	return ast_node->dy.value();
}

bool MouseCoordinates::x_is_relative() const {
	return x().at(0) == '+' || x().at(0) == '-';
}

bool MouseCoordinates::y_is_relative() const {
	return y().at(0) == '+' || y().at(0) == '-';
}

std::string MouseSelectable::where_to_go() const {
	std::string result;
	if (auto p = std::dynamic_pointer_cast<AST::SelectJS>(ast_node->basic_select_expr)) {
		result += "js selection \"";
		result += IR::SelectJS(p, stack).script();
		result += "\"";
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectText>(ast_node->basic_select_expr)) {
		result += "\"";
		result += IR::SelectText(p, stack).text();
		result += "\"";
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectImg>(ast_node->basic_select_expr)) {
		result += "image \"";
		result += IR::SelectImg(p, stack).img_path().generic_string();
		result += "\"";
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectHomm3>(ast_node->basic_select_expr)) {
		result += "HOMM3 object \"";
		result += IR::SelectHomm3(p, stack).id();
		result += "\"";
	} else {
		throw std::runtime_error("Where to go is unapplicable");
	}
	return result;
}

TimeInterval MouseSelectable::timeout() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("timeout", "TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT");
}

std::string SelectJS::script() const {
	return String(ast_node->str, stack).text();
}

fs::path SelectImg::img_path() const {
	fs::path path = String(ast_node->str, stack).text();

	if (path.is_relative()) {
		path = ast_node->begin().file.parent_path() / path;
	}

	return path;
}

std::string SelectHomm3::id() const {
	return String(ast_node->str, stack).text();
}

std::string SelectText::text() const {
	return String(ast_node->str, stack).text();
}

MouseButton MouseHold::button() const {
	return ToMouseButton(ast_node->button.value());
}

bool Plug::is_on() const {
	return ast_node->is_on();
}

std::string PlugFlash::name() const {
	return Id(ast_node->name, stack).value();
}

std::string PlugNIC::name() const {
	return Id(ast_node->name, stack).value();
}

std::string PlugLink::name() const {
	return Id(ast_node->name, stack).value();
}

std::string PlugHostDev::type() const {
	return ast_node->type.value();
}

std::string PlugHostDev::addr() const {
	return String(ast_node->addr, stack).text();
}

fs::path PlugDVD::path() const {
	fs::path path = String(ast_node->path, stack).text();

	if (path.is_relative()) {
		path = ast_node->path->begin().file.parent_path() / path;
	}

	return path;
}

IR::TimeInterval Shutdown::timeout() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("timeout", "TESTO_SHUTDOWN_DEFAULT_TIMEOUT");
}

std::string Exec::interpreter() const {
	return ast_node->process.value();
}

IR::TimeInterval Exec::timeout() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("timeout", "TESTO_EXEC_DEFAULT_TIMEOUT");
}

std::string Exec::script() const {
	return String(ast_node->commands, stack).text();
}

std::string Copy::from() const {
	fs::path from = String(ast_node->from, stack).text();

	if (ast_node->is_to_guest()) {
		if (from.is_relative()) {
			from = ast_node->from->begin().file.parent_path() / from;
		}
	}

	return from.generic_string();
}

std::string Screenshot::destination() const {
	fs::path dest = String(ast_node->destination, stack).text();

	if (dest.is_relative()) {
		dest = ast_node->destination->begin().file.parent_path() / dest;
	}

	return dest.generic_string();
}

std::string Copy::to() const {
	fs::path to = String(ast_node->to, stack).text();

	if (!ast_node->is_to_guest()) {
		if (to.is_relative()) {
			to = ast_node->to->begin().file.parent_path() / to;
		}
	}

	return to.generic_string();
}

TimeInterval Copy::timeout() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("timeout", "TESTO_COPY_DEFAULT_TIMEOUT");
}

bool Copy::nocheck() const {
	return OptionSeq(ast_node->option_seq, stack).has("nocheck");
}

std::string CycleControl::type() const {
	return ast_node->token.value();
}

}
