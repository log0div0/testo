
#pragma once

#include "Pos.hpp"
#include "Token.hpp"
#include <vector>
#include <set>
#include <memory>
#include <functional>

namespace AST {

struct Node {
	Node(const Token& t): t(t) {}
	virtual ~Node() {}
	virtual Pos begin() const {
		return t.begin();
	}
	virtual Pos end() const {
		return t.end();
	}
	virtual operator std::string() const {
		return t.value();
	}

	Token t;
};

//basic unit of expressions - could be double quoted string or a var_ref (variable)
struct String: public Node {
	using Node::Node;

	std::string text() const {
		if (t.type() == Token::category::quoted_string) {
			return t.value().substr(1, t.value().length() - 2);
		} else {
			return t.value().substr(3, t.value().length() - 6);
		}
	}
};

/*
This is a special Node
It incapsulates the case when
Parser expects either String either regular Token.
Semantic later should convert the String to a specific Token if
needed.
*/
struct StringTokenUnion: public Node {
	StringTokenUnion(const Token& token, std::shared_ptr<String> string, Token::category expected_token_type):
		Node(Token(Token::category::string_token_union, "string_token_union", Pos(), Pos())),
		token(token), string(string), expected_token_type(expected_token_type) {}

	Pos begin() const override {
		if (string) {
			return string->begin();
		} else {
			return token.begin();
		}
	}

	Pos end() const override {
		if (string) {
			return string->end();
		} else {
			return token.end();
		}
	}

	operator std::string() const override {
		if (string) {
			return std::string(*string);
		} else {
			return token.value();
		}
	}

	std::string text() const {
		if (string) {
			return string->text();
		} else {
			return token.value();
		}
	}

	Token token;
	std::shared_ptr<String> string;
	Token::category expected_token_type;
};

struct SelectExpr: public Node {
	using Node::Node;
};

//String or SelectQuery. Used only in
//Wait, Check and Click (in future)
struct Selectable: public SelectExpr {
	using SelectExpr::SelectExpr;

	virtual std::string to_string() const = 0;

	operator std::string() const final {
		std::string result;
		if (is_negated()) {
			result += "!";
		}
		result += to_string();
		return result;
	}

	bool is_negated() const {
		return excl_mark.type() == Token::category::exclamation_mark;
	}

	Token excl_mark;
};

//basic unit of expressions - could be double quoted string or a var_ref (variable)
struct SelectJS: public Selectable {
	SelectJS(const Token& js, std::shared_ptr<String> script):
		Selectable(js), script(script) {}

	Pos end() const override {
		return script->end();
	}

	std::string to_string() const override {
		return std::string(*script);
	}

	std::string text() const {
		return script->text();
	}

	std::shared_ptr<String> script;
};

struct SelectText: public Selectable {
	SelectText(std::shared_ptr<String> text):
		Selectable(Token(Token::category::select_text, "select_text", Pos(), Pos())), _text(text) {}

	Pos begin() const override {
		return _text->begin();
	}

	Pos end() const override {
		return _text->end();
	}

	std::string to_string() const override {
		return std::string(*_text);
	}

	std::string text() const {
		return _text->text();
	}

	std::shared_ptr<String> _text;
};

struct SelectImg: public Selectable {
	SelectImg(const Token& img, std::shared_ptr<String> img_path):
		Selectable(img), img_path(img_path) {}

	Pos end() const override {
		return img_path->end();
	}

	std::string to_string() const override {
		return std::string(*img_path);
	}

	std::string text() const {
		return img_path->text();
	}

	std::shared_ptr<String> img_path;
};

struct SelectHomm3: public Selectable {
	SelectHomm3(const Token& homm3, std::shared_ptr<String> id):
		Selectable(homm3), id(id) {}

	Pos end() const override {
		return id->end();
	}

	std::string to_string() const override {
		return std::string(*id);
	}

	std::string text() const {
		return id->text();
	}

	std::shared_ptr<String> id;
};

struct KeyCombination: public Node {
	KeyCombination(const std::vector<Token>& buttons):
		Node(Token(Token::category::key_combination, "key_combination", Pos(), Pos())),
		buttons(buttons) {}

	Pos begin() const {
		return buttons[0].begin();
	}

	Pos end() const {
		return buttons.back().end();
	}

	operator std::string() const {
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
	KeySpec(std::shared_ptr<KeyCombination> combination, std::shared_ptr<StringTokenUnion> times):
		Node(Token(Token::category::key_spec, "key_spec", Pos(), Pos())),
		combination(combination),
		times(times) {}

	Pos begin() const {
		return combination->begin();
	}

	Pos end() const {
		if (times) {
			return times->end();
		} else {
			return combination->end();
		}
	}

	operator std::string() const {
		std::string result = std::string(*combination);
		if (times) {
			result += "*" + std::string(*times);
		}
		return result;
	}

	std::shared_ptr<KeyCombination> combination;
	std::shared_ptr<StringTokenUnion> times;
};

struct Action: public Node {
	using Node::Node;

	Pos end() const override {
		return delim.end();
	}

	virtual std::string to_string() const {
		return t.value();
	}

	operator std::string() const final {
		std::string result = to_string();
		if (delim.type() == Token::category::semi) {
			result += delim.value();
		}
		return result;
	}

	Token delim;
};

struct Empty: public Action {
	Empty(): Action(Token()) {}

	std::string to_string() const override {
		return "";
	}
};

struct Abort: public Action {
	Abort(const Token& abort, std::shared_ptr<String> message):
		Action(abort), message(message) {}

	Pos end() const override {
		return message->end();
	}

	std::string to_string() const override {
		return t.value() + " " + std::string(*message);
	}

	std::shared_ptr<String> message;
};

struct Print: public Action {
	Print(const Token& print, std::shared_ptr<String> message):
		Action(print), message(message) {}

	Pos end() const override {
		return message->end();
	}

	std::string to_string() const override {
		return t.value() + " " + std::string(*message);
	}

	std::shared_ptr<String> message;
};

struct Type: public Action {
	Type(const Token& type, std::shared_ptr<String> text, std::shared_ptr<StringTokenUnion> interval):
		Action(type), text(text), interval(interval) {}

	Pos end() const override {
		if (interval) {
			return interval->end();
		} else {
			return text->end();
		}
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + std::string(*text);

		if (interval) {
			result += std::string(*interval);
		}

		return result;
	}

	std::shared_ptr<String> text;
	std::shared_ptr<StringTokenUnion> interval;
};

struct SelectBinOp: public SelectExpr {
	SelectBinOp(std::shared_ptr<SelectExpr> left, const Token& op, std::shared_ptr<SelectExpr> right):
		SelectExpr(op), left(left), right(right) {}

	Pos begin() const override {
		return left->begin();
	}

	Pos end() const override {
		return right->end();
	}

	operator std::string() const override {
		return std::string(*left) + " " + t.value() + " " + std::string(*right);
	}

	std::shared_ptr<SelectExpr> left;
	std::shared_ptr<SelectExpr> right;
};

struct SelectParentedExpr: public Selectable {
	SelectParentedExpr(const Token& lparen, std::shared_ptr<SelectExpr> select_expr, const Token& rparen):
		Selectable(lparen), select_expr(select_expr), rparen(rparen) {}

	Pos end() const override {
		return rparen.end();
	}

	std::string to_string() const override {
		return t.value() + std::string(*select_expr) + rparen.value();
	}

	std::shared_ptr<SelectExpr> select_expr;
	Token rparen;
};

struct Wait: public Action {
	Wait(const Token& wait, std::shared_ptr<SelectExpr> select_expr, std::shared_ptr<StringTokenUnion> timeout, std::shared_ptr<StringTokenUnion> interval):
		Action(wait), select_expr(select_expr), timeout(timeout), interval(interval) {}

	Pos end() const override {
		if (interval) {
			return interval->end();
		} else if (timeout) {
			return timeout->end();
		} else {
			return select_expr->end();
		}
	}

	std::string to_string() const override {
		std::string result = t.value();
		result += " " + std::string(*select_expr);

		if (timeout) {
			result += " timeout "  + std::string(*timeout);
		}

		if (interval) {
			result += " interval "  + std::string(*interval);
		}

		return result;
	}

	std::shared_ptr<SelectExpr> select_expr;
	std::shared_ptr<StringTokenUnion> timeout;
	std::shared_ptr<StringTokenUnion> interval;
};

struct Sleep: public Action {
	Sleep(const Token& sleep, std::shared_ptr<StringTokenUnion> timeout):
		Action(sleep),  timeout(timeout) {}

	Pos end() const override {
		return timeout->end();
	}

	std::string to_string() const override {
		return t.value() + " for " + std::string(*timeout);
	}


	std::shared_ptr<StringTokenUnion> timeout;
};

struct Press: public Action {
	Press(const Token& press, const std::vector<std::shared_ptr<KeySpec>> keys, std::shared_ptr<StringTokenUnion> interval):
		Action(press), keys(keys), interval(interval) {}

	Pos end() const override {
		if (interval) {
			return interval->end();
		} else {
			return keys[keys.size() - 1]->end();
		}
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + std::string(*keys[0]);

		for (size_t i = 1; i < keys.size(); i++) {
			result += ", " + std::string(*keys[i]);
		}

		if (interval) {
			result += " interval " + std::string(*interval);
		}

		return result;
	}

	std::vector<std::shared_ptr<KeySpec>> keys;
	std::shared_ptr<StringTokenUnion> interval;
};

struct Hold: public Action {
	Hold(const Token& hold, std::shared_ptr<KeyCombination> combination):
		Action(hold), combination(combination) {}

	Pos end() const override {
		return combination->end();
	}

	std::string to_string() const override {
		return t.value() + " " + std::string(*combination);
	}

	std::shared_ptr<KeyCombination> combination;
};

struct Release: public Action {
	Release(const Token& release, std::shared_ptr<KeyCombination> combination):
		Action(release), combination(combination) {}

	Pos end() const override {
		if (combination) {
			return combination->end();
		} else {
			return t.end();
		}
	}

	std::string to_string() const override {
		std::string result = t.value();
		if (combination) {
			result += " " + std::string(*combination);
		}

		return result;
	}

	std::shared_ptr<KeyCombination> combination = nullptr;
};

struct MouseMoveTarget: public Node {
	using Node::Node;
};

struct MouseAdditionalSpecifier: public Node {
	MouseAdditionalSpecifier(const Token& name, const Token& lparen, const Token& arg, const Token& rparen):
		Node(Token(Token::category::mouse_additional_specifier, "mouse_additional_specifier", Pos(), Pos())), name(name), lparen(lparen), arg(arg), rparen(rparen) {}

	Pos begin() const override {
		return name.begin();
	}

	Pos end() const override {
		return rparen.end();
	}

	operator std::string() const override {
		std::string result = "." + name.value() + "(";
		if (arg) {
			result += arg.value();
		}
		result += ")";
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
	MouseSelectable(std::shared_ptr<Selectable> selectable,
		const std::vector<std::shared_ptr<MouseAdditionalSpecifier>>& specifiers,
		std::shared_ptr<StringTokenUnion> timeout):
		MouseMoveTarget(Token(Token::category::mouse_selectable, "mouse_selectable", Pos(), Pos())),
		selectable(selectable), specifiers(specifiers), timeout(timeout) {}

	Pos begin() const override {
		return selectable->begin();
	}

	Pos end() const override {
		if (timeout) {
			return timeout->end();
		} else if (specifiers.size()) {
			return specifiers[specifiers.size() - 1]->end();
		} else {
			return selectable->end();
		}
	}

	operator std::string() const override {
		std::string result = std::string(*selectable);

		for (auto specifier: specifiers) {
			result += std::string(*specifier);
		}

		if (timeout) {
			result += " timeout " + std::string(*timeout);
		}

		return result;
	}

	std::shared_ptr<Selectable> selectable = nullptr;
	std::vector<std::shared_ptr<MouseAdditionalSpecifier>> specifiers;
	std::shared_ptr<StringTokenUnion> timeout;
};

struct MouseCoordinates: public MouseMoveTarget {
	MouseCoordinates(const Token& dx, const Token& dy):
		MouseMoveTarget(Token(Token::category::mouse_coordinates, "mouse_coordinates", Pos(), Pos())), dx(dx), dy(dy) {}

	Pos begin() const override {
		return dx.begin();
	}

	Pos end() const override {
		return dy.end();
	}

	operator std::string() const override {
		return dx.value() + " " + dy.value();;
	}

	std::string text() const {
		return std::string(*this);
	}

	Token dx;
	Token dy;
};

struct MouseEvent: public Node {
	using Node::Node;
};

struct MouseMoveClick: public MouseEvent {
	MouseMoveClick(const Token& event, std::shared_ptr<MouseMoveTarget> object):
		MouseEvent(event), object(object) {}

	Pos end() const override {
		if (object) {
			return object->end();
		} else {
			return t.end();
		}
	}

	operator std::string() const override {
		std::string result = t.value();
		if (object) {
			result += " " + std::string(*object);
		}
		return result;
	}

	std::shared_ptr<MouseMoveTarget> object = nullptr;
};

struct MouseHold: public MouseEvent {
	MouseHold(const Token& hold, const Token& button):
		MouseEvent(hold), button(button) {}

	Pos end() const override {
		return button.end();
	}

	operator std::string() const override {
		return t.value() + " " + button.value();
	}

	Token button;
};

struct MouseRelease: public MouseEvent {
	using MouseEvent::MouseEvent;
};

struct MouseWheel: public MouseEvent {
	MouseWheel(const Token& wheel, const Token& direction):
		MouseEvent(wheel), direction(direction) {}

	Pos end() const override {
		return direction.end();
	}

	operator std::string() const override {
		return t.value() + " " + direction.value();
	}

	Token direction;
};

struct Mouse: public Action {
	Mouse(const Token& mouse, std::shared_ptr<MouseEvent> event):
		Action(mouse), event(event) {}

	Pos end() const override {
		return event->end();
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + std::string(*event);
		return result;
	}

	std::shared_ptr<MouseEvent> event = nullptr;
};

struct PlugResource: public Node {
	using Node::Node;
};

struct PlugNIC: public PlugResource {
	PlugNIC(const Token& nic, std::shared_ptr<StringTokenUnion> name):
		PlugResource(nic), name(name) {}

	Pos end() const override {
		return name->end();
	}

	operator std::string() const override {
		std::string result = t.value() + " " + std::string(*name);
		return result;
	}

	std::shared_ptr<StringTokenUnion> name = nullptr;
};

struct PlugLink: public PlugResource {
	PlugLink(const Token& link, std::shared_ptr<StringTokenUnion> name):
		PlugResource(link), name(name) {}

	Pos end() const override {
		return name->end();
	}

	operator std::string() const override {
		std::string result = t.value() + " " + std::string(*name);
		return result;
	}

	std::shared_ptr<StringTokenUnion> name = nullptr;
};

struct PlugFlash: public PlugResource {
	PlugFlash(const Token& flash, std::shared_ptr<StringTokenUnion> name):
		PlugResource(flash), name(name) {}

	Pos end() const override {
		return name->end();
	}

	operator std::string() const override {
		std::string result = t.value() + " " + std::string(*name);
		return result;
	}

	std::shared_ptr<StringTokenUnion> name = nullptr;
};

struct PlugDVD: public PlugResource {
	PlugDVD(const Token& dvd, std::shared_ptr<String> path):
		PlugResource(dvd), path(path) {}

	Pos end() const override {
		if (path) {
			return path->end();
		} else {
			return begin();
		}
	}

	operator std::string() const override {
		std::string result = t.value();

		if (path) {
			result += " " + std::string(*path);
		}
		return result;
	}

	std::shared_ptr<String> path = nullptr;
};

struct PlugHostDev: public PlugResource {
	PlugHostDev(const Token& hostdev, const Token& type, std::shared_ptr<String> addr):
		PlugResource(hostdev), type(type), addr(addr) {}

	Pos end() const override {
		return addr->end();
	}

	operator std::string() const override {
		std::string result = t.value() + " " + type.value() + " " + std::string(*addr);
		return result;
	}

	Token type;
	std::shared_ptr<String> addr = nullptr;
};

//Also is used for unplug
struct Plug: public Action {
	Plug(const Token& plug, std::shared_ptr<PlugResource> resource):
		Action(plug),
		resource(resource) {}

	Pos end() const override {
		return resource->end();
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + std::string(*resource);
		return result;
	}

	bool is_on() const {
		return (t.type() == Token::category::plug);
	}

	std::shared_ptr<PlugResource> resource;
};


struct Start: public Action {
	using Action::Action;
};

struct Stop: public Action {
	using Action::Action;
};

struct Shutdown: public Action {
	Shutdown(const Token& shutdown, std::shared_ptr<StringTokenUnion> timeout):
		Action(shutdown), timeout(timeout) {}

	Pos end() const override {
		if (timeout) {
			return timeout->end();
		} else {
			return t.end();
		}
	}

	std::string to_string() const override {
		std::string result = t.value();
		if (timeout) {
			result += " timeout " + std::string(*timeout);
		}
		return result;
	}

	std::shared_ptr<StringTokenUnion> timeout;
};

struct Exec: public Action {
	Exec(const Token& exec, const Token& process, std::shared_ptr<String> commands, std::shared_ptr<StringTokenUnion> timeout):
		Action(exec),
		process_token(process),
		commands(commands), timeout(timeout) {}

	Pos end() const override {
		if (timeout) {
			return timeout->end();
		}
		return commands->end();
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + process_token.value() + " " + std::string(*commands);
		if (timeout) {
			result += " timeout " + std::string(*timeout);
		}
		return result;
	}

	Token process_token;
	std::shared_ptr<String> commands;
	std::shared_ptr<StringTokenUnion> timeout;
};

//Now this node holds actions copyto and copyfrom
//Cause they're really similar
struct Copy: public Action {
	Copy(const Token& copy, std::shared_ptr<String> from, std::shared_ptr<String> to, const Token& nocheck, std::shared_ptr<StringTokenUnion> timeout):
		Action(copy),
		from(from),
		to(to), nocheck(nocheck), timeout(timeout) {}

	Pos end() const override {
		if (timeout) {
			return timeout->end();
		}
		return to->end();
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + std::string(*from) + " " + std::string(*to);
		if (timeout) {
			result += " timeout " + std::string(*timeout);
		}
		return result;
	}

	//return true if we copy to guest,
	//false if from guest to host
	bool is_to_guest() const {
		return t.type() == Token::category::copyto;
	}

	std::shared_ptr<String> from;
	std::shared_ptr<String> to;
	Token nocheck;
	std::shared_ptr<StringTokenUnion> timeout;
};

struct Screenshot: public Action {
	Screenshot(const Token& screenshot, std::shared_ptr<String> destination):
		Action(screenshot), destination(destination) {}

	Pos end() const override {
		return destination->end();
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + std::string(*destination);
		return result;
	}

	std::shared_ptr<String> destination = nullptr;
};

struct ActionBlock: public Action {
	ActionBlock(const Token& open_brace, const Token& close_brace, std::vector<std::shared_ptr<Action>> actions):
		Action(Token(Token::category::action_block, "action_block", Pos(), Pos())),
		open_brace(open_brace),
		close_brace(close_brace),
		actions(actions) {}

	Pos begin() const override {
		return open_brace.begin();
	}

	Pos end() const override {
		return close_brace.end();
	}

	std::string to_string() const override {
		std::string result;

		for (auto action: actions) {
			result += std::string(*action);

		}
		return result;
	}

	Token open_brace;
	Token close_brace;
	std::vector<std::shared_ptr<Action>> actions;
};

struct Cmd: public Node {
	using Node::Node;
	operator std::string() const final {
		return to_string();
	}
	virtual std::string to_string() const = 0;
};

struct RegularCmd: public Cmd {
	RegularCmd(std::shared_ptr<StringTokenUnion> entity, std::shared_ptr<Action> action):
		Cmd(Token(Token::category::regular_cmd, "regular_cmd", Pos(), Pos())),
		entity(entity),
		action(action) {}

	Pos begin() const override {
		return entity->begin();
	}

	Pos end() const override {
		return action->end();
	}

	std::string to_string() const override {
		return std::string(*entity) + " " + std::string(*action);
	}

	std::shared_ptr<StringTokenUnion> entity;
	std::shared_ptr<Action> action;
};

struct CmdBlock: public Cmd {
	CmdBlock(const Token& open_brace, const Token& close_brace, std::vector<std::shared_ptr<Cmd>> commands):
		Cmd(Token(Token::category::cmd_block, "cmd_block", Pos(),Pos())),
		open_brace(open_brace),
		close_brace(close_brace),
		commands(commands) {}

	Pos begin() const override {
		return open_brace.begin();
	}

	Pos end() const override {
		return close_brace.end();
	}

	std::string to_string() const override {
		std::string result;

		for (auto command: commands) {
			result += std::string(*command);

		}
		return result;
	}

	Token open_brace;
	Token close_brace;
	std::vector<std::shared_ptr<Cmd>> commands;
};

//High-level constructions
//may be machine, flash, macro or test declaration
struct Stmt: public Node {
	using Node::Node;
	operator std::string() const final {
		return to_string();
	}
	virtual std::string to_string() const = 0;
};

//Used only in macro-tests
struct StmtBlock: public Stmt {
	StmtBlock(const Token& open_brace, const Token& close_brace, std::vector<std::shared_ptr<Stmt>> stmts):
		Stmt(Token(Token::category::stmt_block, "stmt_block", Pos(), Pos())),
		open_brace(open_brace),
		close_brace(close_brace),
		stmts(stmts) {}

	Pos begin() const override {
		return open_brace.begin();
	}

	Pos end() const override {
		return close_brace.end();
	}

	std::string to_string() const override {
		std::string result;

		for (auto stmt: stmts) {
			result += std::string(*stmt);

		}
		return result;
	}

	Token open_brace;
	Token close_brace;
	std::vector<std::shared_ptr<Stmt>> stmts;
};

struct MacroArg: public Node {
	MacroArg(const Token& name, std::shared_ptr<String> default_value):
		Node(name), default_value(default_value) {}

	Pos end() const override {
		if (default_value) {
			return default_value->end();
		} else {
			return t.end();
		}
	}

	operator std::string() const override {
		std::string result = t.value();
		if (default_value) {
			result += "=" + std::string(*default_value);
		}
		return result;
	}

	std::string name() const {
		return t.value();
	}

	std::shared_ptr<String> default_value = nullptr;
};

struct IMacroBody: public Node {
	using Node::Node;
};

//IfClause if also an action
template <typename MacroBodyType>
struct MacroBody: public IMacroBody {
	MacroBody(std::shared_ptr<MacroBodyType> macro_body):
		IMacroBody(macro_body->t),
		macro_body(macro_body) {}

	Pos begin() const {
		return macro_body->begin();
	}

	Pos end() const {
		return macro_body->end();
	}

	operator std::string() const {
		std::string result = std::string(*macro_body);
		return result;
	}

	std::shared_ptr<MacroBodyType> macro_body;
};

struct MacroBodyStmt: public Node {
	static const std::string desc() { return "statements"; }
	using BlockType = AST::StmtBlock;

	MacroBodyStmt(std::shared_ptr<StmtBlock> stmt_block):
		Node(stmt_block->t), stmt_block(stmt_block) {}

	Pos begin() const {
		return stmt_block->begin();
	}

	Pos end() const {
		return stmt_block->end();
	}

	operator std::string() const {
		std::string result = std::string(*stmt_block);
		return result;
	}

	std::shared_ptr<AST::StmtBlock> stmt_block;
};


struct MacroBodyCommand: public Node {
	static const std::string desc() { return "commands"; }
	using BlockType = AST::CmdBlock;

	MacroBodyCommand(std::shared_ptr<CmdBlock> cmd_block):
		Node(cmd_block->t), cmd_block(cmd_block) {}

	Pos begin() const {
		return cmd_block->begin();
	}

	Pos end() const {
		return cmd_block->end();
	}

	operator std::string() const {
		std::string result = std::string(*cmd_block);
		return result;
	}

	std::shared_ptr<CmdBlock> cmd_block;
};

struct MacroBodyAction: public Node {
	static const std::string desc() { return "actions"; }
	using BlockType = AST::ActionBlock;

	MacroBodyAction(std::shared_ptr<ActionBlock> action_block):
		Node(action_block->t), action_block(action_block) {}

	Pos begin() const {
		return action_block->begin();
	}

	Pos end() const {
		return action_block->end();
	}

	operator std::string() const {
		std::string result = std::string(*action_block);
		return result;
	}

	std::shared_ptr<ActionBlock> action_block;
};

struct Macro: public Stmt {
	Macro(const Token& macro,
		const Token& name,
		const std::vector<std::shared_ptr<MacroArg>>& args,
		const std::vector<Token>& body_tokens):
			Stmt(macro), name(name), args(args),
			body_tokens(body_tokens) {}

	Pos end() const override {
		return body_tokens.back().end();
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + name.value() + "(";
		for (auto arg: args) {
			result += std::string(*arg) + " ,";
		}
		result += ")";

		for (auto t: body_tokens) {
			result += " ";
			result += t.value();
		}

		return result;
	}

	Token name;
	std::vector<std::shared_ptr<MacroArg>> args;
	std::shared_ptr<IMacroBody> body = nullptr;
	std::vector<Token> body_tokens;
};

struct IMacroCall {
	IMacroCall(const std::vector<std::shared_ptr<String>>& args_): args(args_) {}
	virtual ~IMacroCall() {}

	virtual Token name() const = 0;
	virtual Pos begin() const = 0;
	virtual Pos end() const = 0;

	std::vector<std::shared_ptr<String>> args;
};

template <typename BaseType>
struct MacroCall: public BaseType, IMacroCall {
	MacroCall(const Token& macro_name, const std::vector<std::shared_ptr<String>>& args):
		BaseType(macro_name), IMacroCall(args) {}

	std::string to_string() const override {
		std::string result = t.value() + ("(");
		for (auto arg: args) {
			result += std::string(*arg) + " ,";
		}
		result += ")";
		return result;
	}

	Token name() const override {
		return t;
	}

	Pos IMacroCall::begin() const override {
		return BaseType::begin();
	}

	Pos IMacroCall::end() const override {
		return BaseType::end();
	}
};

struct IAttrValue: public Node {
	using Node::Node;

	virtual Token::category type() const = 0;
};

template <typename AttrType>
struct AttrValue: public IAttrValue {
	AttrValue(std::shared_ptr<AttrType> attr_value):
		IAttrValue(attr_value->t),
		attr_value(attr_value) {}

	Pos begin() const {
		return attr_value->begin();
	}

	Pos end() const {
		return attr_value->end();
	}

	operator std::string() const {
		return std::string(*attr_value);
	}

	Token::category type() const {
		return attr_value->type();
	}

	std::shared_ptr<AttrType> attr_value;
};

struct SimpleAttr: public Node {
	SimpleAttr(std::shared_ptr<StringTokenUnion> _value):
		Node(Token(Token::category::simple_attr, "", _value->begin(), _value->end())), value(_value) {}

	Pos begin() const {
		return value->begin();
	}

	Pos end() const {
		return value->end();
	}

	operator std::string() const {
		return std::string(*value);
	}

	Token::category type() const {
		return value->expected_token_type;
	}

	std::shared_ptr<StringTokenUnion> value;
};

struct Attr: public Node {
	Attr(const Token& name, const Token& id, std::shared_ptr<IAttrValue> value):
		Node(Token(Token::category::attr, "", Pos(), Pos())),
		name(name),
		id(id),
		value(value) {}

	Pos begin() const {
		return name.begin();
	}

	Pos end() const {
		return value->end();
	}

	operator std::string() const {
		return name.value() + ": " + std::string(*value);
	}

	Token name;
	Token id;
	std::shared_ptr<IAttrValue> value;
};

struct AttrBlock: public Node {
	AttrBlock(const Token& open_brace, const Token& close_brace, std::vector<std::shared_ptr<Attr>> attrs):
		Node(Token(Token::category::attr_block, "", Pos(),Pos())),
		open_brace(open_brace),
		close_brace(close_brace),
		attrs(attrs) {}

	Pos begin() const {
		return open_brace.begin();
	}

	Pos end() const {
		return close_brace.end();
	}

	operator std::string() const {
		std::string result;

		for (auto attr: attrs) {
			result += std::string(*attr);

		}
		return result;
	}

	Token::category type() const {
		return t.type();
	}

	Token open_brace;
	Token close_brace;
	std::vector<std::shared_ptr<Attr>> attrs;
};

struct Test: public Stmt {
	Test(std::shared_ptr<AttrBlock> attrs,
		const Token& test, std::shared_ptr<StringTokenUnion> name,
		const std::vector<std::shared_ptr<StringTokenUnion>>& parents,
		std::shared_ptr<CmdBlock> cmd_block):
		Stmt(test),
		attrs(attrs),
		name(name),
		parents(parents),
		cmd_block(cmd_block) {}

	Pos begin() const override {
		if (attrs) {
			return attrs->begin();
		} else {
			return t.begin();
		}
	}

	Pos end() const override {
		return cmd_block->end();
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + std::string(*name);
		return result; //for now
	}

	std::shared_ptr<AttrBlock> attrs;
	std::shared_ptr<StringTokenUnion> name;
	std::vector<std::shared_ptr<StringTokenUnion>> parents;
	std::shared_ptr<CmdBlock> cmd_block;
};

struct Controller: public Stmt {
	Controller(const Token& controller, std::shared_ptr<StringTokenUnion> name, std::shared_ptr<AttrBlock> attr_block):
		Stmt(controller),
		name(name),
		attr_block(attr_block) {}

	Pos end() const override {
		return attr_block->end();
	}

	std::string to_string() const override {
		return t.value() + " " + std::string(*name) + " " + std::string(*attr_block);
	}

	std::shared_ptr<StringTokenUnion> name;
	std::shared_ptr<AttrBlock> attr_block;
};

struct Param: public Stmt {
	Param(const Token& param_token, const Token& name, std::shared_ptr<String> value):
		Stmt(param_token), name(name), value(value) {}

	Pos end() const override {
		return value->end();
	}

	std::string to_string() const override {
		std::string result = t.value() + " " + name.value() + " " + std::string(*value);
		return result;
	}

	Token name;
	std::shared_ptr<String> value;
};

struct Program: public Node {
	Program (const std::vector<std::shared_ptr<Stmt>> stmts):
		Node(Token(Token::category::program, "program", Pos(),Pos())),
		stmts(stmts) {}

	Pos begin() const override {
		return stmts[0]->begin();
	}

	Pos end() const override {
		return stmts[stmts.size() - 1]->end();
	}

	operator std::string() const override {
		std::string result;

		for (auto stmt: stmts) {
			result += std::string(*stmt);

		}

		return result;
	}

	std::vector<std::shared_ptr<Stmt>> stmts;
};

struct Expr: public Node {
	using Node::Node;
};

struct StringExpr: public Expr {
	StringExpr(const std::shared_ptr<String> str_): Expr(str_->t), str(str_) {}

	operator std::string() const override {
		return *str;
	}

	std::shared_ptr<String> str;
};

struct Negation: public Expr {
	Negation(const Token& not_token, std::shared_ptr<Expr> expr_): Expr(not_token), expr(expr_) {}

	operator std::string() const override {
		return "NOT " + (std::string)(*expr);
	}

	std::shared_ptr<Expr> expr;
};

struct Defined: public Expr {
	Defined(const Token& defined, const Token& var):
		Expr(defined), var(var) {}

	Pos end() const override {
		return var.end();
	}

	operator std::string() const override {
		return t.value() + " " + var.value();
	}

	Token var;
};

struct Comparison: public Expr {
	Comparison(const Token& op, std::shared_ptr<String> left, std::shared_ptr<String> right):
		Expr(op), left(left), right(right) {}

	Pos begin() const override {
		return left->begin();
	}

	Pos end() const override{
		return right->end();
	}

	operator std::string() const override {
		return std::string(*left) + " " + t.value() + " " + std::string(*right);
	}

	Token op() const {
		return t;
	}

	std::shared_ptr<String> left;
	std::shared_ptr<String> right;
};

struct Check: public Expr {
	Check(const Token& check, std::shared_ptr<SelectExpr> select_expr, std::shared_ptr<StringTokenUnion> timeout, std::shared_ptr<StringTokenUnion> interval):
		Expr(check), select_expr(select_expr), timeout(timeout), interval(interval) {}

	Pos end() const override {
		if (interval) {
			return interval->end();
		} else if (timeout) {
			return timeout->end();
		} else {
			return select_expr->end();
		}
	}

	operator std::string() const override {
		std::string result = t.value();
		result += " " + std::string(*select_expr);

		if (timeout) {
			result += " timeout " + std::string(*timeout);
		}

		if (interval) {
			result += " interval " + std::string(*interval);
		}

		return result;
	}

	std::shared_ptr<SelectExpr> select_expr;

	std::shared_ptr<StringTokenUnion> timeout;
	std::shared_ptr<StringTokenUnion> interval;
};

struct ParentedExpr: public Expr {
	ParentedExpr(const Token& lparen, std::shared_ptr<Expr> expr, const Token& rparen):
		Expr(lparen), expr(expr), rparen(rparen) {}

	Pos end() const override {
		return rparen.end();
	}

	operator std::string() const override {
		return t.value() + std::string(*expr) + rparen.value();
	}

	std::shared_ptr<Expr> expr;
	Token rparen;
};

struct BinOp: public Expr {
	BinOp(const Token& op, std::shared_ptr<Expr> left, std::shared_ptr<Expr> right):
		Expr(op), left(left), right(right) {}

	Pos begin() const override {
		return left->begin();
	}

	Pos end() const override {
		return right->end();
	}

	operator std::string() const override {
		return std::string("BINOP: ") + std::string(*left) + " " + t.value() + " " + std::string(*right);
	}

	Token op() const {
		return t;
	}

	std::shared_ptr<Expr> left;
	std::shared_ptr<Expr> right;
};

struct IfClause: public Action {
	IfClause(const Token& if_token, const Token& open_paren, std::shared_ptr<Expr> expr,
		const Token& close_paren, std::shared_ptr<Action> if_action, const Token& else_token,
		std::shared_ptr<Action> else_action):
		Action(if_token),
		open_paren(open_paren),
		expr(expr),
		close_paren(close_paren),
		if_action(if_action),
		else_token(else_token),
		else_action(else_action)
	{}

	Pos end() const override {
		if (has_else()) {
			return else_action->end();
		} else {
			return if_action->end();
		}
	}

	std::string to_string() const override {
		std::string result;

		result += t.value() + " " +
			open_paren.value() + std::string(*expr) +
			close_paren.value() + " " + std::string(*if_action);

		if (has_else()) {
			result += std::string("\n") + else_token.value() + " " +  std::string(*else_action);
		}

		return result;
	}

	bool has_else() const {
		return else_token;
	}

	Token open_paren;
	std::shared_ptr<Expr> expr;
	Token close_paren;
	std::shared_ptr<Action> if_action;
	Token else_token;
	std::shared_ptr<Action> else_action;
};

struct CounterList: public Node {
	using Node::Node;
};

struct Range: public CounterList {
	Range(const Token& range, std::shared_ptr<StringTokenUnion> r1, std::shared_ptr<StringTokenUnion> r2):
		CounterList(range), r1(r1), r2(r2) {}

	Pos begin() const override {
		return r1->begin();
	}

	Pos end() const override {
		if (r2) {
			return r2->end();
		} else {
			return r1->end();
		}
	}

	operator std::string() const override {
		std::string result = t.value() + " " + std::string(*r1);
		if (r2) {
			result += " " + std::string(*r2);
		}
		return result;
	}

	std::shared_ptr<StringTokenUnion> r1 = nullptr;
	std::shared_ptr<StringTokenUnion> r2 = nullptr;
};

struct ForClause: public Action {
	ForClause(const Token& for_token, const Token& counter,	std::shared_ptr<CounterList> counter_list,
		std::shared_ptr<Action> cycle_body, const Token& else_token,
		std::shared_ptr<Action> else_action):
		Action(for_token),
		counter(counter),
		counter_list(counter_list),
		cycle_body(cycle_body),
		else_token(else_token),
		else_action(else_action) {}

	Pos end() const override {
		if (else_token) {
			return else_action->end();
		} else {
			return cycle_body->end();
		}
	}

	std::string to_string() const override {
		std::string result = t.value() + "(" + counter.value() + " IN " + std::string(*counter_list) + ")";

		result += std::string(*cycle_body);
		if (else_action) {
			result += std::string(*else_action);
		}
		return result;
	}

	Token counter;
	std::shared_ptr<CounterList> counter_list = nullptr;
	std::shared_ptr<Action> cycle_body;

	Token else_token;
	std::shared_ptr<Action> else_action;
};

struct CycleControl: public Action {
	using Action::Action;
};

}

