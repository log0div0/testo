
#pragma once

#include "Pos.hpp"
#include "Token.hpp"
#include <vector>
#include <set>
#include <memory>
#include <functional>

namespace AST {

struct Node {
	virtual ~Node() {}
	virtual Pos begin() const = 0;
	virtual Pos end() const = 0;
	virtual std::string to_string() const = 0;
};

struct String: public Node {
	String(Token token_): token(std::move(token_)) {}

	Pos begin() const override {
		return token.begin();
	}
	Pos end() const override {
		return token.end();
	}
	std::string to_string() const override {
		return token.value();
	}

	std::string text() const {
		if (token.type() == Token::category::quoted_string) {
			return token.value().substr(1, token.value().length() - 2);
		} else {
			return token.value().substr(3, token.value().length() - 6);
		}
	}

	Token token;
};

template <Token::category category>
struct ISingleToken: public Node {
	static std::shared_ptr<ISingleToken> from_string(const std::string& str);
};

template <Token::category category>
struct SingleToken: public ISingleToken<category> {
	SingleToken(Token token_): token(std::move(token_)) {}

	Pos begin() const override {
		return token.begin();
	}
	Pos end() const override {
		return token.end();
	}
	std::string to_string() const override {
		return token.value();
	}

	Token token;
};

template <typename T>
struct Unparsed: public T {
	Unparsed(std::shared_ptr<String> string_): string(std::move(string_)) {}

	Pos begin() const override {
		return string->begin();
	}
	Pos end() const override {
		return string->end();
	}
	std::string to_string() const override {
		return string->to_string();
	}

	std::shared_ptr<String> string;
};

using Number = ISingleToken<Token::category::number>;
using TimeInterval = ISingleToken<Token::category::time_interval>;
using Id = ISingleToken<Token::category::id>;
using QuotedString = ISingleToken<Token::category::quoted_string>;
using Boolean = ISingleToken<Token::category::boolean>;
using Size = ISingleToken<Token::category::size>;

struct SelectExpr: public Node {
};

struct SelectSimpleExpr: public SelectExpr {
};

struct SelectNegationExpr: public SelectSimpleExpr {
	SelectNegationExpr(Token excl_mark_, std::shared_ptr<SelectSimpleExpr> expr_):
		excl_mark(std::move(excl_mark_)), expr(std::move(expr_)) {}

	Pos begin() const override {
		return excl_mark.begin();
	}

	Pos end() const override {
		return expr->end();
	}

	std::string to_string() const override {
		return excl_mark.value() + expr->to_string();
	}

	Token excl_mark;
	std::shared_ptr<SelectSimpleExpr> expr;
};

struct BasicSelectExpr: public SelectSimpleExpr {
	BasicSelectExpr(Token token_, std::shared_ptr<String> str_):
		token(std::move(token_)), str(std::move(str_)) {}

	Pos begin() const override {
		if (token) {
			return token.begin();
		}
		return str->begin();
	}

	Pos end() const override {
		return str->end();
	}

	std::string to_string() const override {
		if (token) {
			return token.value() + " " + str->to_string();
		}
		return str->to_string();
	}

	std::string text() const {
		return str->text();
	}

	Token token;
	std::shared_ptr<String> str;
};

struct SelectJS: public BasicSelectExpr {
	using BasicSelectExpr::BasicSelectExpr;
};

struct SelectText: public BasicSelectExpr {
	using BasicSelectExpr::BasicSelectExpr;
};

struct SelectImg: public BasicSelectExpr {
	using BasicSelectExpr::BasicSelectExpr;
};

struct SelectHomm3: public BasicSelectExpr {
	using BasicSelectExpr::BasicSelectExpr;
};

struct SelectBinOp: public SelectExpr {
	SelectBinOp(std::shared_ptr<SelectExpr> left_, Token op_, std::shared_ptr<SelectExpr> right_):
		left(std::move(left_)), op(std::move(op_)), right(std::move(right_)) {}

	Pos begin() const override {
		return left->begin();
	}

	Pos end() const override {
		return right->end();
	}

	std::string to_string() const override {
		return left->to_string() + " " + op.value() + " " + right->to_string();
	}

	std::shared_ptr<SelectExpr> left;
	Token op;
	std::shared_ptr<SelectExpr> right;
};

struct SelectParentedExpr: public SelectSimpleExpr {
	SelectParentedExpr(Token lparen_, std::shared_ptr<SelectExpr> select_expr_, Token rparen_):
		lparen(std::move(lparen_)), select_expr(std::move(select_expr_)), rparen(std::move(rparen_)) {}

	Pos begin() const override {
		return lparen.begin();
	}

	Pos end() const override {
		return rparen.end();
	}

	std::string to_string() const override {
		return lparen.value() + select_expr->to_string() + rparen.value();
	}

	Token lparen;
	std::shared_ptr<SelectExpr> select_expr;
	Token rparen;
};

struct IKeyCombination: public Node {
	static std::shared_ptr<IKeyCombination> from_string(const std::string& str);
};

struct KeyCombination: public IKeyCombination {
	KeyCombination(std::vector<Token> buttons_):
		buttons(std::move(buttons_)) {}

	Pos begin() const override {
		return buttons.front().begin();
	}

	Pos end() const override {
		return buttons.back().end();
	}

	std::string to_string() const override {
		auto btns = get_buttons();
		std::string result = btns[0];
		for (size_t i = 1; i < btns.size(); i++) {
			result += "+" + btns[i];
		}

		return result;
	}

	std::vector<std::string> get_buttons() const {
		std::vector<std::string> result;

		for (auto& button: buttons) {
			std::string button_str = button.value();
			std::transform(button_str.begin(), button_str.end(), button_str.begin(), ::toupper);
			result.push_back(button_str);
		}

		return result;
	}

	std::vector<Token> buttons;
};

struct KeySpec: public Node {
	KeySpec(std::shared_ptr<IKeyCombination> combination_, std::shared_ptr<Number> times_):
		combination(std::move(combination_)),
		times(std::move(times_)) {}

	Pos begin() const override {
		return combination->begin();
	}

	Pos end() const override {
		if (times) {
			return times->end();
		} else {
			return combination->end();
		}
	}

	std::string to_string() const override {
		std::string result = combination->to_string();
		if (times) {
			result += "*" + times->to_string();
		}
		return result;
	}

	std::shared_ptr<IKeyCombination> combination;
	std::shared_ptr<Number> times;
};

struct Option: public Node {
	Option(Token name_): name(std::move(name_)) {}

	Pos begin() const override {
		return name.begin();
	}

	Pos end() const override {
		if (value) {
			return value->end();
		}
		return name.end();
	}

	std::string to_string() const override {
		if (value) {
			return name.value() + " " + value->to_string();
		}
		return name.value();
	}

	Token name;
	std::shared_ptr<Node> value;
};

struct OptionSeq: public Node {
	Pos begin() const override {
		if (options.size()) {
			return options.front()->begin();
		}
		return {};
	}

	Pos end() const override {
		if (options.size()) {
			return options.back()->end();
		}
		return {};
	}

	std::string to_string() const override {
		std::string res;
		for (size_t i = 0; i < options.size(); ++i) {
			if (i) {
				res += " ";
			}
			res += options[i]->to_string();
		}
		return res;
	}

	size_t size() const {
		return options.size();
	}

	std::shared_ptr<Option> get(const std::string& option_name) const {
		for (auto& option: options) {
			if (option->name.value() == option_name) {
				return option;
			}
		}
		return nullptr;
	}

	std::vector<std::shared_ptr<Option>> options;
};

struct Action: public Node {
	static const std::string desc() { return "actions"; }
};

struct ActionWithDelim: public Action {
	ActionWithDelim(std::shared_ptr<Action> action_, Token delim_):
		action(std::move(action_)), delim(std::move(delim_)) {}

	Pos begin() const override {
		return action->begin();
	}

	Pos end() const override {
		return delim.end();
	}

	std::string to_string() const override {
		std::string result = action->to_string();
		if (delim.type() == Token::category::semi) {
			result += delim.value();
		}
		return result;
	}

	std::shared_ptr<Action> action;
	Token delim;
};

struct Empty: public Action {
	Pos begin() const override {
		return {};
	}
	Pos end() const override {
		return {};
	}
	std::string to_string() const override {
		return {};
	}
};

struct Abort: public Action {
	Abort(Token abort_, std::shared_ptr<String> message_):
		abort(std::move(abort_)), message(std::move(message_)) {}

	Pos begin() const override {
		return abort.begin();
	}

	Pos end() const override {
		return message->end();
	}

	std::string to_string() const override {
		return abort.value() + " " + message->to_string();
	}

	Token abort;
	std::shared_ptr<String> message;
};

struct Print: public Action {
	Print(Token print_, std::shared_ptr<String> message_):
		print(std::move(print_)), message(std::move(message_)) {}

	Pos begin() const override {
		return print.begin();
	}

	Pos end() const override {
		return message->end();
	}

	std::string to_string() const override {
		return print.value() + " " + message->to_string();
	}

	Token print;
	std::shared_ptr<String> message;
};

struct Type: public Action {
	Type(Token type_, std::shared_ptr<String> text_, std::shared_ptr<OptionSeq> option_seq_):
		type(std::move(type_)), text(std::move(text_)), option_seq(std::move(option_seq_)) {}

	Pos begin() const override {
		return type.begin();
	}

	Pos end() const override {
		if (option_seq->size()) {
			return option_seq->end();
		} else {
			return text->end();
		}
	}

	std::string to_string() const override {
		std::string result = type.value() + " " + text->to_string();
		if (option_seq->size()) {
			result += " " + option_seq->to_string();
		}
		return result;
	}

	Token type;
	std::shared_ptr<String> text;
	std::shared_ptr<OptionSeq> option_seq;
};

struct Wait: public Action {
	Wait(Token wait_, std::shared_ptr<SelectExpr> select_expr_, std::shared_ptr<OptionSeq> option_seq_):
		wait(std::move(wait_)), select_expr(std::move(select_expr_)), option_seq(std::move(option_seq_)) {}

	Pos begin() const override {
		return wait.begin();
	}

	Pos end() const override {
		if (option_seq->size()) {
			return option_seq->end();
		} else {
			return select_expr->end();
		}
	}

	std::string to_string() const override {
		std::string result = wait.value() + " " + select_expr->to_string();
		if (option_seq->size()) {
			result += " " + option_seq->to_string();
		}
		return result;
	}

	Token wait;
	std::shared_ptr<SelectExpr> select_expr;
	std::shared_ptr<OptionSeq> option_seq;
};

struct Sleep: public Action {
	Sleep(Token sleep_, std::shared_ptr<TimeInterval> timeout_):
		sleep(std::move(sleep_)), timeout(std::move(timeout_)) {}

	Pos begin() const override {
		return sleep.begin();
	}

	Pos end() const override {
		return timeout->end();
	}

	std::string to_string() const override {
		return sleep.value() + " for " + timeout->to_string();
	}

	Token sleep;
	std::shared_ptr<TimeInterval> timeout;
};

struct Press: public Action {
	Press(Token press_, std::vector<std::shared_ptr<KeySpec>> keys_, std::shared_ptr<OptionSeq> option_seq_):
		press(std::move(press_)), keys(std::move(keys_)), option_seq(std::move(option_seq_)) {}

	Pos begin() const override {
		return press.begin();
	}

	Pos end() const override {
		if (option_seq->size()) {
			return option_seq->end();
		} else {
			return keys[keys.size() - 1]->end();
		}
	}

	std::string to_string() const override {
		std::string result = press.value() + " " + keys[0]->to_string();

		for (size_t i = 1; i < keys.size(); i++) {
			result += ", " + keys[i]->to_string();
		}

		if (option_seq->size()) {
			result += " " + option_seq->to_string();
		}

		return result;
	}

	Token press;
	std::vector<std::shared_ptr<KeySpec>> keys;
	std::shared_ptr<OptionSeq> option_seq;
};

struct Hold: public Action {
	Hold(Token hold_, std::shared_ptr<IKeyCombination> combination_):
		hold(std::move(hold_)), combination(std::move(combination_)) {}

	Pos begin() const override {
		return hold.begin();
	}

	Pos end() const override {
		return combination->end();
	}

	std::string to_string() const override {
		return hold.value() + " " + combination->to_string();
	}

	Token hold;
	std::shared_ptr<IKeyCombination> combination;
};

struct Release: public Action {
	Release(Token release_, std::shared_ptr<IKeyCombination> combination_):
		release(std::move(release_)), combination(std::move(combination_)) {}

	Pos begin() const override {
		return release.begin();
	}

	Pos end() const override {
		if (combination) {
			return combination->end();
		} else {
			return release.end();
		}
	}

	std::string to_string() const override {
		std::string result = release.value();
		if (combination) {
			result += " " + combination->to_string();
		}
		return result;
	}

	Token release;
	std::shared_ptr<IKeyCombination> combination;
};

struct MouseMoveTarget: public Node {
};

struct MouseAdditionalSpecifier: public Node {
	MouseAdditionalSpecifier(Token name_, Token lparen_, Token arg_, Token rparen_):
		name(std::move(name_)), lparen(std::move(lparen_)), arg(std::move(arg_)), rparen(std::move(rparen_)) {}

	Pos begin() const override {
		return name.begin();
	}

	Pos end() const override {
		return rparen.end();
	}

	std::string to_string() const override {
		std::string result = "." + name.value() + lparen.value();
		if (arg) {
			result += arg.value();
		}
		result += rparen.value();
		return result;
	}

	bool is_from() const {
		return (name.value() == "from_top" ||
			name.value() == "from_bottom" ||
			name.value() == "from_left" ||
			name.value() == "from_right");
	}

	bool is_centering() const {
		return (name.value() == "left_bottom" ||
			name.value() == "left_center" ||
			name.value() == "left_top" ||
			name.value() == "center_bottom" ||
			name.value() == "center" ||
			name.value() == "center_top" ||
			name.value() == "right_bottom" ||
			name.value() == "right_center" ||
			name.value() == "right_top");
	}

	bool is_moving() const {
		return (name.value() == "move_left" ||
			name.value() == "move_right" ||
			name.value() == "move_up" ||
			name.value() == "move_down");
	}

	Token name;
	Token lparen;
	Token arg;
	Token rparen;
};

struct MouseSelectable: public MouseMoveTarget {
	MouseSelectable(
		std::shared_ptr<BasicSelectExpr> basic_select_expr_,
		std::vector<std::shared_ptr<MouseAdditionalSpecifier>> mouse_additional_specifiers_,
		std::shared_ptr<OptionSeq> option_seq_
	):
		basic_select_expr(std::move(basic_select_expr_)),
		mouse_additional_specifiers(std::move(mouse_additional_specifiers_)),
		option_seq(std::move(option_seq_))
	{}

	Pos begin() const override {
		return basic_select_expr->begin();
	}

	Pos end() const override {
		if (option_seq->size()) {
			return option_seq->end();
		} else if (mouse_additional_specifiers.size()) {
			return mouse_additional_specifiers[mouse_additional_specifiers.size() - 1]->end();
		} else {
			return basic_select_expr->end();
		}
	}

	std::string to_string() const override {
		std::string result = basic_select_expr->to_string();

		for (auto mouse_additional_specifier: mouse_additional_specifiers) {
			result += mouse_additional_specifier->to_string();
		}

		if (option_seq->size()) {
			result += " " + option_seq->to_string();
		}

		return result;
	}

	std::shared_ptr<BasicSelectExpr> basic_select_expr;
	std::vector<std::shared_ptr<MouseAdditionalSpecifier>> mouse_additional_specifiers;
	std::shared_ptr<OptionSeq> option_seq;
};

struct MouseCoordinates: public MouseMoveTarget {
	MouseCoordinates(Token dx_, Token dy_): dx(std::move(dx_)), dy(std::move(dy_)) {}

	Pos begin() const override {
		return dx.begin();
	}

	Pos end() const override {
		return dy.end();
	}

	std::string to_string() const override {
		return text();
	}

	std::string text() const {
		return dx.value() + " " + dy.value();;
	}

	Token dx;
	Token dy;
};

struct MouseEvent: public Node {
};

struct MouseMoveClick: public MouseEvent {
	MouseMoveClick(Token event_, std::shared_ptr<MouseMoveTarget> object_):
		event(std::move(event_)), object(std::move(object_)) {}

	Pos begin() const override {
		return event.begin();
	}

	Pos end() const override {
		if (object) {
			return object->end();
		} else {
			return event.end();
		}
	}

	std::string to_string() const override {
		std::string result = event.value();
		if (object) {
			result += " " + object->to_string();
		}
		return result;
	}

	Token event;
	std::shared_ptr<MouseMoveTarget> object;
};

struct MouseHold: public MouseEvent {
	MouseHold(Token hold_, Token button_):
		hold(std::move(hold_)), button(std::move(button_)) {}

	Pos begin() const override {
		return hold.begin();
	}

	Pos end() const override {
		return button.end();
	}

	std::string to_string() const override {
		return hold.value() + " " + button.value();
	}

	Token hold;
	Token button;
};

struct MouseRelease: public MouseEvent {
	MouseRelease(Token release_):
		release(std::move(release_)){}

	Pos begin() const override {
		return release.begin();
	}

	Pos end() const override {
		return release.end();
	}

	std::string to_string() const override {
		return release.value();
	}

	Token release;
};

struct MouseWheel: public MouseEvent {
	MouseWheel(Token wheel_, Token direction_):
		wheel(std::move(wheel_)), direction(std::move(direction_)) {}

	Pos begin() const override {
		return wheel.begin();
	}

	Pos end() const override {
		return direction.end();
	}

	std::string to_string() const override {
		return wheel.value() + " " + direction.value();
	}

	Token wheel;
	Token direction;
};

struct Mouse: public Action {
	Mouse(Token mouse_, std::shared_ptr<MouseEvent> event_):
		mouse(std::move(mouse_)), event(std::move(event_)) {}

	Pos begin() const override {
		return mouse.begin();
	}

	Pos end() const override {
		return event->end();
	}

	std::string to_string() const override {
		return mouse.value() + " " + event->to_string();
	}

	Token mouse;
	std::shared_ptr<MouseEvent> event;
};

struct PlugResource: public Node {
	using Node::Node;
};

struct PlugResourceWithName: PlugResource {
	PlugResourceWithName(Token type_, std::shared_ptr<Id> name_):
		type(std::move(type_)), name(std::move(name_)) {}

	Pos begin() const override {
		return type.begin();
	}

	Pos end() const override {
		return name->end();
	}

	std::string to_string() const override {
		return type.value() + " " + name->to_string();
	}

	Token type;
	std::shared_ptr<Id> name;
};

struct PlugNIC: public PlugResourceWithName {
	using PlugResourceWithName::PlugResourceWithName;
};

struct PlugLink: public PlugResourceWithName {
	using PlugResourceWithName::PlugResourceWithName;
};

struct PlugFlash: public PlugResourceWithName {
	using PlugResourceWithName::PlugResourceWithName;
};

struct PlugDVD: public PlugResource {
	PlugDVD(Token dvd_, std::shared_ptr<String> path_):
		dvd(std::move(dvd_)), path(std::move(path_)) {}

	Pos begin() const override {
		return dvd.begin();
	}

	Pos end() const override {
		if (path) {
			return path->end();
		} else {
			return begin();
		}
	}

	std::string to_string() const override {
		std::string result = dvd.value();
		if (path) {
			result += " " + path->to_string();
		}
		return result;
	}

	Token dvd;
	std::shared_ptr<String> path;
};

struct PlugHostDev: public PlugResource {
	PlugHostDev(Token hostdev_, Token type_, std::shared_ptr<String> addr_):
		hostdev(std::move(hostdev_)), type(std::move(type_)), addr(std::move(addr_)) {}

	Pos begin() const override {
		return hostdev.begin();
	}

	Pos end() const override {
		return addr->end();
	}

	std::string to_string() const override {
		return hostdev.value() + " " + type.value() + " " + addr->to_string();
	}

	Token hostdev;
	Token type;
	std::shared_ptr<String> addr;
};

//Also is used for unplug
struct Plug: public Action {
	Plug(Token plug_, std::shared_ptr<PlugResource> resource_):
		plug(std::move(plug_)), resource(std::move(resource_)) {}

	Pos begin() const override {
		return plug.begin();
	}

	Pos end() const override {
		return resource->end();
	}

	std::string to_string() const override {
		return plug.value() + " " + resource->to_string();
	}

	bool is_on() const {
		return plug.type() == Token::category::plug;
	}

	Token plug;
	std::shared_ptr<PlugResource> resource;
};

struct ElementaryAction: Action {
	ElementaryAction(Token token_):
		token(std::move(token_)){}

	Pos begin() const override {
		return token.begin();
	}

	Pos end() const override {
		return token.end();
	}

	std::string to_string() const override {
		return token.value();
	}

	Token token;
};

struct Start: public ElementaryAction {
	using ElementaryAction::ElementaryAction;
};

struct Stop: public ElementaryAction {
	using ElementaryAction::ElementaryAction;
};

struct CycleControl: public ElementaryAction {
	using ElementaryAction::ElementaryAction;
};

struct Shutdown: public Action {
	Shutdown(Token shutdown_, std::shared_ptr<OptionSeq> option_seq_):
		shutdown(std::move(shutdown_)), option_seq(std::move(option_seq_)) {}

	Pos begin() const override {
		return shutdown.begin();
	}

	Pos end() const override {
		if (option_seq->size()) {
			return option_seq->end();
		} else {
			return shutdown.end();
		}
	}

	std::string to_string() const override {
		std::string result = shutdown.value();
		if (option_seq->size()) {
			result += " " + option_seq->to_string();
		}
		return result;
	}

	Token shutdown;
	std::shared_ptr<OptionSeq> option_seq;
};

struct Exec: public Action {
	Exec(Token exec_, Token process_, std::shared_ptr<String> commands_, std::shared_ptr<OptionSeq> option_seq_):
		exec(std::move(exec_)),
		process(std::move(process_)),
		commands(std::move(commands_)),
		option_seq(std::move(option_seq_)) {}

	Pos begin() const override {
		return exec.begin();
	}

	Pos end() const override {
		if (option_seq->size()) {
			return option_seq->end();
		}
		return commands->end();
	}

	std::string to_string() const override {
		std::string result = exec.value() + " " + process.value() + " " + commands->to_string();
		if (option_seq->size()) {
			result += " " + option_seq->to_string();
		}
		return result;
	}

	Token exec;
	Token process;
	std::shared_ptr<String> commands;
	std::shared_ptr<OptionSeq> option_seq;
};

//Now this node holds actions copyto and copyfrom
//Cause they're really similar
struct Copy: public Action {
	Copy(Token copy_, std::shared_ptr<String> from_, std::shared_ptr<String> to_, std::shared_ptr<OptionSeq> option_seq_):
		copy(std::move(copy_)),
		from(std::move(from_)),
		to(std::move(to_)),
		option_seq(std::move(option_seq_)) {}

	Pos begin() const override {
		return copy.begin();
	}

	Pos end() const override {
		if (option_seq->size()) {
			return option_seq->end();
		}
		return to->end();
	}

	std::string to_string() const override {
		std::string result = copy.value() + " " + from->to_string() + " " + to->to_string();
		if (option_seq->size()) {
			result += " " + option_seq->to_string();
		}
		return result;
	}

	//return true if we copy to guest,
	//false if from guest to host
	bool is_to_guest() const {
		return copy.type() == Token::category::copyto;
	}

	Token copy;
	std::shared_ptr<String> from;
	std::shared_ptr<String> to;
	std::shared_ptr<OptionSeq> option_seq;
};

struct Screenshot: public Action {
	Screenshot(Token screenshot_, std::shared_ptr<String> destination_):
		screenshot(std::move(screenshot_)), destination(std::move(destination_)) {}

	Pos begin() const override {
		return screenshot.begin();
	}

	Pos end() const override {
		return destination->end();
	}

	std::string to_string() const override {
		return screenshot.value() + " " + destination->to_string();
	}

	Token screenshot;
	std::shared_ptr<String> destination;
};

struct IBlock {
	virtual ~IBlock() {}
};

template <typename Item>
struct Block: public Item, public IBlock {
	Block(Token open_brace_, Token close_brace_, std::vector<std::shared_ptr<Item>> items_):
		open_brace(std::move(open_brace_)),
		close_brace(std::move(close_brace_)),
		items(std::move(items_)) {}

	Pos begin() const override {
		return open_brace.begin();
	}

	Pos end() const override {
		return close_brace.end();
	}

	std::string to_string() const override {
		std::string result = "{ ";

		for (auto action: items) {
			result += action->to_string();

		}

		result += " }";
		return result;
	}

	Token open_brace;
	Token close_brace;
	std::vector<std::shared_ptr<Item>> items;
};

struct Cmd: public Node {
	static const std::string desc() { return "commands"; }
};

struct RegularCmd: public Cmd {
	RegularCmd(std::shared_ptr<Id> entity_, std::shared_ptr<Action> action_):
		entity(std::move(entity_)),
		action(std::move(action_)) {}

	Pos begin() const override {
		return entity->begin();
	}

	Pos end() const override {
		return action->end();
	}

	std::string to_string() const override {
		return entity->to_string() + " " + action->to_string();
	}

	std::shared_ptr<Id> entity;
	std::shared_ptr<Action> action;
};

//High-level constructions
//may be machine, flash, macro or test declaration
struct Stmt: public Node {
	static const std::string desc() { return "statements"; }
};

struct MacroArg: public Node {
	MacroArg(Token name_token_, std::shared_ptr<String> default_value_):
		name_token(std::move(name_token_)), default_value(std::move(default_value_)) {}

	Pos begin() const override {
		return name_token.begin();
	}

	Pos end() const override {
		if (default_value) {
			return default_value->end();
		} else {
			return name_token.end();
		}
	}

	std::string to_string() const override {
		std::string result = name_token.value();
		if (default_value) {
			result += "=" + default_value->to_string();
		}
		return result;
	}

	std::string name() const {
		return name_token.value();
	}

	Token name_token;
	std::shared_ptr<String> default_value;
};

struct Macro: public Stmt {
	Macro(Token macro_,
		Token name_,
		std::vector<std::shared_ptr<MacroArg>> args_,
		std::vector<Token> body_tokens_):
			macro(std::move(macro_)), name(std::move(name_)), args(std::move(args_)),
			body_tokens(std::move(body_tokens_)) {}

	Pos begin() const override {
		return macro.begin();
	}

	Pos end() const override {
		return body_tokens.back().end();
	}

	std::string to_string() const override {
		std::string result = macro.value() + " " + name.value() + "(";
		for (auto arg: args) {
			result += arg->to_string() + " ,";
		}
		result += ")";

		for (auto body_token: body_tokens) {
			result += " ";
			result += body_token.value();
		}

		return result;
	}

	Token macro;
	Token name;
	std::vector<std::shared_ptr<MacroArg>> args;
	std::vector<Token> body_tokens;

	std::shared_ptr<IBlock> block;
};

struct IMacroCall {
	IMacroCall(Token name_, Token lparen_, std::vector<std::shared_ptr<String>> args_, Token rparen_):
		name(std::move(name_)), lparen(std::move(lparen_)), args(std::move(args_)), rparen(std::move(rparen_)) {}
	virtual ~IMacroCall() {}

	Pos begin() const {
		return name.begin();
	}

	Pos end() const {
		return rparen.end();
	}

	std::string to_string() const {
		std::string result = name.value() + ("(");
		for (size_t i = 0; i < args.size(); ++i) {
			if (i) {
				result += ", ";
			}
			result += args[i]->to_string();
		}
		result += ")";
		return result;
	}

	Token name;
	Token lparen;
	std::vector<std::shared_ptr<String>> args;
	Token rparen;
};

template <typename BaseType>
struct MacroCall: public BaseType, IMacroCall {
	MacroCall(Token name, Token lparen, std::vector<std::shared_ptr<String>> args, Token rparen):
		IMacroCall(std::move(name), std::move(lparen), std::move(args), std::move(rparen)) {}

	Pos begin() const override {
		return IMacroCall::begin();
	}

	Pos end() const override {
		return IMacroCall::end();
	}

	std::string to_string() const override {
		return IMacroCall::to_string();
	}
};

struct Attr: public Node {
	Attr(Token name_token_, Token id_, std::shared_ptr<Node> value_):
		name_token(std::move(name_token_)),
		id(std::move(id_)),
		value(std::move(value_)) {}

	Pos begin() const override {
		return name_token.begin();
	}

	Pos end() const override {
		return value->end();
	}

	std::string to_string() const override {
		return name_token.value() + ": " + value->to_string();
	}

	std::string name() const {
		return name_token.value();
	}

	Token name_token;
	Token id;
	std::shared_ptr<Node> value;
};

struct AttrBlock: public Node {
	AttrBlock(Token open_brace_, Token close_brace_, std::vector<std::shared_ptr<Attr>> attrs_):
		open_brace(std::move(open_brace_)),
		close_brace(std::move(close_brace_)),
		attrs(std::move(attrs_)) {}

	Pos begin() const override {
		return open_brace.begin();
	}

	Pos end() const override {
		return close_brace.end();
	}

	std::string to_string() const override {
		std::string result;
		for (auto attr: attrs) {
			result += attr->to_string();

		}
		return result;
	}

	Token open_brace;
	Token close_brace;
	std::vector<std::shared_ptr<Attr>> attrs;
};

struct Test: public Stmt {
	Test(std::shared_ptr<AttrBlock> attrs_,
		Token test_,
		std::shared_ptr<Id> name_,
		std::vector<std::shared_ptr<Id>> parents_,
		std::shared_ptr<AST::Block<AST::Cmd>> cmd_block_
	):
		attrs(std::move(attrs_)),
		test(std::move(test_)),
		name(std::move(name_)),
		parents(std::move(parents_)),
		cmd_block(std::move(cmd_block_)) {}

	Pos begin() const override {
		if (attrs) {
			return attrs->begin();
		} else {
			return test.begin();
		}
	}

	Pos end() const override {
		return cmd_block->end();
	}

	std::string to_string() const override {
		return test.value() + " " + name->to_string();
	}

	std::shared_ptr<AttrBlock> attrs;
	Token test;
	std::shared_ptr<Id> name;
	std::vector<std::shared_ptr<Id>> parents;
	std::shared_ptr<AST::Block<AST::Cmd>> cmd_block;
};

struct Controller: public Stmt {
	Controller(Token controller_, std::shared_ptr<Id> name_, std::shared_ptr<AttrBlock> attr_block_):
		controller(std::move(controller_)),
		name(std::move(name_)),
		attr_block(std::move(attr_block_)) {}

	Pos begin() const override {
		return controller.begin();
	}

	Pos end() const override {
		return attr_block->end();
	}

	std::string to_string() const override {
		return controller.value() + " " + name->to_string() + " " + attr_block->to_string();
	}

	Token controller;
	std::shared_ptr<Id> name;
	std::shared_ptr<AttrBlock> attr_block;
};

struct Param: public Stmt {
	Param(Token param_token_, Token name_, std::shared_ptr<String> value_):
		param_token(std::move(param_token_)), name(std::move(name_)), value(std::move(value_)) {}

	Pos begin() const override {
		return param_token.begin();
	}

	Pos end() const override {
		return value->end();
	}

	std::string to_string() const override {
		return param_token.value() + " " + name.value() + " " + value->to_string();
	}

	Token param_token;
	Token name;
	std::shared_ptr<String> value;
};

struct Program: public Node {
	Program(std::vector<std::shared_ptr<Stmt>> stmts_):
		stmts(std::move(stmts_)) {}

	Pos begin() const override {
		return stmts[0]->begin();
	}

	Pos end() const override {
		return stmts[stmts.size() - 1]->end();
	}

	std::string to_string() const override {
		std::string result;

		for (auto stmt: stmts) {
			result += stmt->to_string();
		}

		return result;
	}

	std::vector<std::shared_ptr<Stmt>> stmts;
};

struct Expr: public Node {
};

struct SimpleExpr: public Expr {
};

struct StringExpr: public SimpleExpr {
	StringExpr(std::shared_ptr<String> str_): str(std::move(str_)) {}

	Pos begin() const override {
		return str->begin();
	}

	Pos end() const override {
		return str->end();
	}

	std::string to_string() const override {
		return str->to_string();
	}

	std::shared_ptr<String> str;
};

struct Negation: public SimpleExpr {
	Negation(Token not_token_, std::shared_ptr<SimpleExpr> expr_):
		not_token(std::move(not_token_)), expr(std::move(expr_)) {}

	Pos begin() const override {
		return not_token.begin();
	}

	Pos end() const override {
		return expr->end();
	}

	std::string to_string() const override {
		return not_token.value() + expr->to_string();
	}

	Token not_token;
	std::shared_ptr<SimpleExpr> expr;
};

struct Defined: public SimpleExpr {
	Defined(Token defined_, Token var_):
		defined(std::move(defined_)), var(std::move(var_)) {}

	Pos begin() const override {
		return defined.begin();
	}

	Pos end() const override {
		return var.end();
	}

	std::string to_string() const override {
		return defined.value() + " " + var.value();
	}

	Token defined;
	Token var;
};

struct Comparison: public Expr {
	Comparison(Token op_, std::shared_ptr<String> left_, std::shared_ptr<String> right_):
		op(std::move(op_)), left(std::move(left_)), right(std::move(right_)) {}

	Pos begin() const override {
		return left->begin();
	}

	Pos end() const override{
		return right->end();
	}

	std::string to_string() const override {
		return left->to_string() + " " + op.value() + " " + right->to_string();
	}

	Token op;
	std::shared_ptr<String> left;
	std::shared_ptr<String> right;
};

struct BinOp: public Expr {
	BinOp(Token op_, std::shared_ptr<Expr> left_, std::shared_ptr<Expr> right_):
		op(std::move(op_)), left(std::move(left_)), right(std::move(right_)) {}

	Pos begin() const override {
		return left->begin();
	}

	Pos end() const override{
		return right->end();
	}

	std::string to_string() const override {
		return left->to_string() + " " + op.value() + " " + right->to_string();
	}

	Token op;
	std::shared_ptr<Expr> left;
	std::shared_ptr<Expr> right;
};

struct Check: public SimpleExpr {
	Check(Token check_, std::shared_ptr<SelectExpr> select_expr_, std::shared_ptr<OptionSeq> option_seq_):
		check(std::move(check_)), select_expr(std::move(select_expr_)), option_seq(std::move(option_seq_)) {}

	Pos begin() const override {
		return check.begin();
	}

	Pos end() const override {
		if (option_seq->size()) {
			return option_seq->end();
		} else {
			return select_expr->end();
		}
	}

	std::string to_string() const override {
		std::string result = check.value();
		result += " " + select_expr->to_string();

		if (option_seq->size()) {
			result += " " + option_seq->to_string();
		}

		return result;
	}

	Token check;
	std::shared_ptr<SelectExpr> select_expr;
	std::shared_ptr<OptionSeq> option_seq;
};

struct ParentedExpr: public SimpleExpr {
	ParentedExpr(Token lparen_, std::shared_ptr<Expr> expr_, Token rparen_):
		lparen(std::move(lparen_)), expr(std::move(expr_)), rparen(std::move(rparen_)) {}

	Pos begin() const override {
		return lparen.begin();
	}

	Pos end() const override {
		return rparen.end();
	}

	std::string to_string() const override {
		return lparen.value() + expr->to_string() + rparen.value();
	}

	Token lparen;
	std::shared_ptr<Expr> expr;
	Token rparen;
};

struct IfClause: public Action {
	IfClause(Token if_token_, Token open_paren_, std::shared_ptr<Expr> expr_, Token close_paren_,
		std::shared_ptr<Action> if_action_,
		Token else_token_,
		std::shared_ptr<Action> else_action_):
		if_token(std::move(if_token_)), open_paren(std::move(open_paren_)), expr(std::move(expr_)), close_paren(std::move(close_paren_)),
		if_action(std::move(if_action_)),
		else_token(std::move(else_token_)),
		else_action(std::move(else_action_))
	{}

	Pos begin() const override {
		return if_token.begin();
	}

	Pos end() const override {
		if (has_else()) {
			return else_action->end();
		} else {
			return if_action->end();
		}
	}

	std::string to_string() const override {
		std::string result;

		result += if_token.value() + " " +
			open_paren.value() + expr->to_string() +
			close_paren.value() + " " + if_action->to_string();

		if (has_else()) {
			result += std::string("\n") + else_token.value() + " " +  else_action->to_string();
		}

		return result;
	}

	bool has_else() const {
		return else_token;
	}

	Token if_token;
	Token open_paren;
	std::shared_ptr<Expr> expr;
	Token close_paren;
	std::shared_ptr<Action> if_action;
	Token else_token;
	std::shared_ptr<Action> else_action;
};

struct CounterList: public Node {
};

struct Range: public CounterList {
	Range(Token range_, std::shared_ptr<Number> r1_, std::shared_ptr<Number> r2_):
		range(std::move(range_)), r1(std::move(r1_)), r2(std::move(r2_)) {}

	Pos begin() const override {
		return range.begin();
	}

	Pos end() const override {
		if (r2) {
			return r2->end();
		} else {
			return r1->end();
		}
	}

	std::string to_string() const override {
		std::string result = range.value() + " " + r1->to_string();
		if (r2) {
			result += " " + r2->to_string();
		}
		return result;
	}

	Token range;
	std::shared_ptr<Number> r1;
	std::shared_ptr<Number> r2;
};

struct ForClause: public Action {
	ForClause(Token for_token_, Token counter_, std::shared_ptr<CounterList> counter_list_,
		std::shared_ptr<Action> cycle_body_,
		Token else_token_,
		std::shared_ptr<Action> else_action_):
		for_token(std::move(for_token_)), counter(std::move(counter_)), counter_list(std::move(counter_list_)),
		cycle_body(std::move(cycle_body_)),
		else_token(std::move(else_token_)),
		else_action(std::move(else_action_)) {}

	Pos begin() const override {
		return for_token.begin();
	}

	Pos end() const override {
		if (else_token) {
			return else_action->end();
		} else {
			return cycle_body->end();
		}
	}

	std::string to_string() const override {
		std::string result = for_token.value() + "(" + counter.value() + " IN " + counter_list->to_string() + ")";

		result += cycle_body->to_string();
		if (else_action) {
			result += else_action->to_string();
		}
		return result;
	}

	Token for_token;
	Token counter;
	std::shared_ptr<CounterList> counter_list;
	std::shared_ptr<Action> cycle_body;
	Token else_token;
	std::shared_ptr<Action> else_action;
};

}

