
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
	virtual Pos begin() const = 0;
	virtual Pos end() const = 0;
	virtual operator std::string() const = 0;

	Token t;
};

//basic unit of expressions - could be double quoted string or a var_ref (variable)
struct String: public Node {
	String(const Token& string):
		Node(string) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return t.end();
	}

	operator std::string() const {
		return t.value();
	}

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

	Pos begin() const {
		if (string) {
			return string->begin();
		} else {
			return token.begin();
		}
	}

	Pos end() const {
		if (string) {
			return string->end();
		} else {
			return token.end();
		}
	}

	operator std::string() const {
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

//basic unit of expressions - could be double quoted string or a var_ref (variable)
struct SelectJS: public Node {
	SelectJS(const Token& js, std::shared_ptr<String> script):
		Node(js), script(script) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return script->end();
	}

	operator std::string() const {
		return std::string(*script);
	}

	std::string text() const {
		return script->text();
	}

	std::shared_ptr<String> script;
};

//String or SelectQuery. Used only in
//Wait, Check and Click (in future)
struct ISelectable: public Node {
	using Node::Node;

	virtual std::string text() const = 0;
};

template <typename SelectableType>
struct Selectable: public ISelectable {
	Selectable(std::shared_ptr<SelectableType> selectable):
		ISelectable(selectable->t),
		selectable(selectable) {}

	Pos begin() const {
		return selectable->begin();
	}

	Pos end() const {
		return selectable->end();
	}

	operator std::string() const {
		return text();
	}

	std::string text() const {
		return selectable->text();
	}

	std::shared_ptr<SelectableType> selectable;
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
	KeySpec(std::shared_ptr<KeyCombination> combination, const Token& times):
		Node(Token(Token::category::key_spec, "key_spec", Pos(), Pos())),
		combination(combination),
		times(times) {}

	Pos begin() const {
		return combination->begin();
	}

	Pos end() const {
		if (times) {
			return times.end();
		} else {
			return combination->end();
		}
	}

	operator std::string() const {
		std::string result = std::string(*combination);
		if (times) {
			result += "*" + times.value();
		}
		return result;
	}

	uint32_t get_times() const {
		if (times) {
			return std::stoul(times.value());
		} else {
			return 1;
		}
	}

	std::shared_ptr<KeyCombination> combination;
	Token times;
};

struct IAction: public Node {
	using Node::Node;

	virtual void set_delim (const Token& delim)  = 0;
};

//IfClause if also an action
template <typename ActionType>
struct Action: public IAction {
	Action(std::shared_ptr<ActionType> action):
		IAction(action->t),
		action(action) {}

	Pos begin() const {
		return action->begin();
	}

	Pos end() const {
		return delim.end();
	}

	operator std::string() const {
		std::string result = std::string(*action);
		if (delim.type() == Token::category::semi) {
			result += delim.value();
		}
		return result;
	}

	void set_delim (const Token& delim) {
		this->delim = delim;
	}

	std::shared_ptr<ActionType> action;
	Token delim;
};

struct Empty: public Node {
	Empty(): Node(Token()) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return t.end();
	}

	operator std::string() const {
		return "";
	}
};

struct Abort: public Node {
	Abort(const Token& abort, std::shared_ptr<String> message):
		Node(abort), message(message) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return message->end();
	}

	operator std::string() const {
		return t.value() + " " + std::string(*message);
	}

	std::shared_ptr<String> message;
};

struct Print: public Node {
	Print(const Token& print, std::shared_ptr<String> message):
		Node(print), message(message) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return message->end();
	}

	operator std::string() const {
		return t.value() + " " + std::string(*message);
	}

	std::shared_ptr<String> message;
};

struct Type: public Node {
	Type(const Token& type, std::shared_ptr<String> text, std::shared_ptr<StringTokenUnion> interval):
		Node(type), text(text), interval(interval) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (interval) {
			return interval->end();
		} else {
			return text->end();
		}
	}

	operator std::string() const {
		std::string result = t.value() + " " + std::string(*text);

		if (interval) {
			result += std::string(*interval);
		}

		return result;
	}

	std::shared_ptr<String> text;
	std::shared_ptr<StringTokenUnion> interval = nullptr;
};

struct ISelectExpr: public Node {
	using Node::Node;
};

template <typename SelectExprType>
struct SelectExpr: public ISelectExpr {
	SelectExpr(std::shared_ptr<SelectExprType> select_expr):
		ISelectExpr(select_expr->t),
		select_expr(select_expr) {}

	Pos begin() const {
		return select_expr->begin();
	}

	Pos end() const {
		return select_expr->end();
	}

	operator std::string() const {
		return std::string(*(select_expr));
	}

	std::shared_ptr<SelectExprType> select_expr;
};

struct SelectUnOp: public Node {
	SelectUnOp(const Token& op, std::shared_ptr<ISelectExpr> select_expr):
		Node(op), select_expr(select_expr) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return select_expr->end();
	}

	operator std::string() const {
		return t.value() + std::string(*select_expr);
	}

	std::shared_ptr<ISelectExpr> select_expr;
};

struct SelectBinOp: public Node {
	SelectBinOp(std::shared_ptr<ISelectExpr> left, const Token& op, std::shared_ptr<ISelectExpr> right):
		Node(op), left(left), right(right) {}

	Pos begin() const {
		return left->begin();
	}

	Pos end() const {
		return right->end();
	}

	operator std::string() const {
		return std::string(*left) + " " + t.value() + " " + std::string(*right);
	}

	std::shared_ptr<ISelectExpr> left;
	std::shared_ptr<ISelectExpr> right;
};

struct SelectParentedExpr: public Node {
	SelectParentedExpr(const Token& lparen, std::shared_ptr<ISelectExpr> select_expr, const Token& rparen):
		Node(lparen), select_expr(select_expr), rparen(rparen) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return rparen.end();
	}

	operator std::string() const {
		return t.value() + std::string(*select_expr) + rparen.value();
	}

	std::shared_ptr<ISelectExpr> select_expr;
	Token rparen;
};

struct Wait: public Node {
	Wait(const Token& wait, std::shared_ptr<ISelectExpr> select_expr, std::shared_ptr<StringTokenUnion> timeout, std::shared_ptr<StringTokenUnion> interval):
		Node(wait), select_expr(select_expr), timeout(timeout), interval(interval) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (interval) {
			return interval->end();
		} else if (timeout) {
			return timeout->end();
		} else {
			return select_expr->end();
		}
	}

	operator std::string() const {
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

	std::shared_ptr<ISelectExpr> select_expr;
	std::shared_ptr<StringTokenUnion> timeout = nullptr;
	std::shared_ptr<StringTokenUnion> interval = nullptr;
};

struct Sleep: public Node {
	Sleep(const Token& sleep, std::shared_ptr<StringTokenUnion> timeout):
		Node(sleep),  timeout(timeout) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return timeout->end();
	}

	operator std::string() const {
		return t.value() + " for " + std::string(*timeout);
	}

	std::shared_ptr<StringTokenUnion> timeout;
};

struct Press: public Node {
	Press(const Token& press, const std::vector<std::shared_ptr<KeySpec>> keys, std::shared_ptr<StringTokenUnion> interval):
		Node(press), keys(keys), interval(interval) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (interval) {
			return interval->end();
		} else {
			return keys[keys.size() - 1]->end();
		}
	}

	operator std::string() const {
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

struct Hold: public Node {
	Hold(const Token& hold, std::shared_ptr<KeyCombination> combination):
		Node(hold), combination(combination) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return combination->end();
	}

	operator std::string() const {
		return t.value() + " " + std::string(*combination);
	}

	std::shared_ptr<KeyCombination> combination;
};

struct Release: public Node {
	Release(const Token& release, std::shared_ptr<KeyCombination> combination):
		Node(release), combination(combination) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (combination) {
			return combination->end();
		} else {
			return t.end();
		}
	}

	operator std::string() const {
		std::string result = t.value();
		if (combination) {
			result += " " + std::string(*combination);
		}
		
		return result;
	}

	std::shared_ptr<KeyCombination> combination = nullptr;
};

struct IMouseMoveTarget: public Node {
	using Node::Node;

	virtual std::string text() const = 0;
};

template <typename MouseMoveTargetType>
struct MouseMoveTarget: public IMouseMoveTarget {
	MouseMoveTarget(std::shared_ptr<MouseMoveTargetType> target):
		IMouseMoveTarget(target->t),
		target(target) {}

	Pos begin() const {
		return target->begin();
	}

	Pos end() const {
		return target->end();
	}

	operator std::string() const {
		return std::string(*(target));
	}

	std::string text() const {
		return target->text();
	}

	std::shared_ptr<MouseMoveTargetType> target;
};

struct MouseAdditionalSpecifier: public Node {
	MouseAdditionalSpecifier(const Token& name, const Token& lparen, const Token& arg, const Token& rparen):
		Node(Token(Token::category::mouse_additional_specifier, "mouse_additional_specifier", Pos(), Pos())), name(name), lparen(lparen), arg(arg), rparen(rparen) {}

	Pos begin() const {
		return name.begin();
	}

	Pos end() const {
		return rparen.end();
	}

	operator std::string() const {
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

struct MouseSelectable: public Node {
	MouseSelectable(std::shared_ptr<ISelectable> selectable,
		const std::vector<std::shared_ptr<MouseAdditionalSpecifier>>& specifiers,
		std::shared_ptr<StringTokenUnion> timeout): 
		Node(Token(Token::category::mouse_selectable, "mouse_selectable", Pos(), Pos())),
		selectable(selectable), specifiers(specifiers), timeout(timeout) {}

	Pos begin() const {
		return selectable->begin(); 
	}

	Pos end() const {
		if (timeout) {
			return timeout->end();
		} else if (specifiers.size()) {
			return specifiers[specifiers.size() - 1]->end();
		} else {
			return selectable->end();
		}
	}

	operator std::string() const {
		std::string result = std::string(*selectable);

		for (auto specifier: specifiers) {
			result += std::string(*specifier);
		}

		if (timeout) {
			result += " timeout " + std::string(*timeout);
		}

		return result;
	}

	std::string text() const {
		return selectable->text();
	}

	std::shared_ptr<ISelectable> selectable = nullptr;
	std::vector<std::shared_ptr<MouseAdditionalSpecifier>> specifiers;
	std::shared_ptr<StringTokenUnion> timeout = nullptr;
};

struct MouseCoordinates: public Node {
	MouseCoordinates(const Token& dx, const Token& dy):
		Node(Token(Token::category::mouse_coordinates, "mouse_coordinates", Pos(), Pos())), dx(dx), dy(dy) {}

	Pos begin() const {
		return dx.begin();
	}

	Pos end() const {
		return dy.end();
	}

	operator std::string() const {
		return dx.value() + " " + dy.value();;
	}

	std::string text() const {
		return std::string(*this);
	}

	Token dx;
	Token dy;
};

struct IMouseEvent: public Node {
	using Node::Node;
};

template <typename MouseEventType>
struct MouseEvent: public IMouseEvent {
	MouseEvent(std::shared_ptr<MouseEventType> event):
		IMouseEvent(event->t),
		event(event) {}

	Pos begin() const {
		return event->begin();
	}

	Pos end() const {
		return event->end();
	}

	operator std::string() const {
		return std::string(*(event));
	}

	std::shared_ptr<MouseEventType> event;
};


struct MouseMoveClick: public Node {
	MouseMoveClick(const Token& event, std::shared_ptr<IMouseMoveTarget> object):
		Node(event), object(object) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (object) {
			return object->end();
		} else {
			return t.end();
		}
	}

	operator std::string() const {
		std::string result = t.value();
		if (object) {
			result += " " + std::string(*object);
		}
		return result;
	}

	std::shared_ptr<IMouseMoveTarget> object = nullptr;
};

struct MouseHold: public Node {
	MouseHold(const Token& hold, const Token& button):
		Node(hold), button(button) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return button.end();
	}

	operator std::string() const {
		return t.value() + " " + button.value();
	}

	Token button;
};

struct MouseRelease: public Node {
	MouseRelease(const Token& release):
		Node(release) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return t.end();
	}

	operator std::string() const {
		return t.value();
	}
};

struct MouseWheel: public Node {
	MouseWheel(const Token& wheel, const Token& direction):
		Node(wheel), direction(direction) {}

	Pos begin() const {
		return t.end();
	}

	Pos end() const {
		return direction.end();
	}

	operator std::string() const {
		return t.value() + " " + direction.value();
	}

	Token direction;
};

struct Mouse: public Node {
	Mouse(const Token& mouse, std::shared_ptr<IMouseEvent> event):
		Node(mouse), event(event) {}

	Pos begin() const {
		return t.end();
	}

	Pos end() const {
		return event->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + std::string(*event);
		return result;
	}

	std::shared_ptr<IMouseEvent> event = nullptr;
};

//Also is used for unplug
struct Plug: public Node {
	Plug(const Token& plug, const Token& type, std::shared_ptr<StringTokenUnion> name, std::shared_ptr<String> path):
		Node(plug),
		type(type),
		name(name),
		path(path) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (name) {
			return name->end();
		} else {
			return path->end();
		}
	}

	operator std::string() const {
		std::string result = t.value() + " " + type.value() + " ";
		if (name) {
			result += std::string(*name);
		} else {
			result += std::string(*path);
		}
		return result;
	}

	bool is_on() const {
		return (t.type() == Token::category::plug);
	}

	Token type; //nic or flash or dvd
	std::shared_ptr<StringTokenUnion> name; //name of resource to be plugged/unplugged
	std::shared_ptr<String> path; //used only for dvd
};


struct Start: public Node {
	Start(const Token& start):
		Node(start) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return t.end();
	}

	operator std::string() const {
		return t.value();
	}
};

struct Stop: public Node {
	Stop(const Token& stop):
		Node(stop) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return t.end();
	}

	operator std::string() const {
		return t.value();
	}
};

struct Shutdown: public Node {
	Shutdown(const Token& shutdown, std::shared_ptr<StringTokenUnion> timeout):
		Node(shutdown), timeout(timeout) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (timeout) {
			return timeout->end();
		} else {
			return t.end();
		}
	}

	operator std::string() const {
		std::string result = t.value();
		if (timeout) {
			result += " timeout " + std::string(*timeout);
		}
		return result;
	}

	std::shared_ptr<StringTokenUnion> timeout = nullptr;
};

struct Exec: public Node {
	Exec(const Token& exec, const Token& process, std::shared_ptr<String> commands, std::shared_ptr<StringTokenUnion> timeout):
		Node(exec),
		process_token(process),
		commands(commands), timeout(timeout) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (timeout) {
			return timeout->end();
		}
		return commands->end();
	}

	operator std::string() const {
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
struct Copy: public Node {
	Copy(const Token& copy, std::shared_ptr<String> from, std::shared_ptr<String> to, std::shared_ptr<StringTokenUnion> timeout):
		Node(copy),
		from(from),
		to(to), timeout(timeout) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (timeout) {
			return timeout->end();
		}
		return to->end();
	}

	operator std::string() const {
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
	std::shared_ptr<StringTokenUnion> timeout = nullptr;
};

struct ActionBlock: public Node {
	ActionBlock(const Token& open_brace, const Token& close_brace, std::vector<std::shared_ptr<IAction>> actions):
		Node(Token(Token::category::action_block, "action_block", Pos(), Pos())),
		open_brace(open_brace),
		close_brace(close_brace),
		actions(actions) {}

	Pos begin() const {
		return open_brace.begin();
	}

	Pos end() const {
		return close_brace.end();
	}

	operator std::string() const {
		std::string result;

		for (auto action: actions) {
			result += std::string(*action);

		}
		return result;
	}

	Token open_brace;
	Token close_brace;
	std::vector<std::shared_ptr<IAction>> actions;
};

struct Cmd: public Node {
	Cmd(const std::vector<Token>& vms, std::shared_ptr<IAction> action):
		Node(Token(Token::category::cmd, "cmd", Pos(), Pos())),
		vms(vms),
		action(action) {}

	Pos begin() const {
		return vms.begin()->begin();
	}

	Pos end() const {
		return action->end();
	}

	operator std::string() const {
		std::string result = vms.begin()->value();
		for (size_t i = 1; i < vms.size() - 1; i++) {
			result += ", ";
			result += vms[i].value();
		}
		result += " " + std::string(*action);
		return result;
	}

	std::vector<Token> vms;
	std::shared_ptr<IAction> action;
};

struct CmdBlock: public Node {
	CmdBlock(const Token& open_brace, const Token& close_brace, std::vector<std::shared_ptr<Cmd>> commands):
		Node(Token(Token::category::cmd_block, "cmd_block", Pos(),Pos())),
		open_brace(open_brace),
		close_brace(close_brace),
		commands(commands) {}

	Pos begin() const {
		return open_brace.begin();
	}

	Pos end() const {
		return close_brace.end();
	}

	operator std::string() const {
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
struct IStmt: public Node {
	using Node::Node;
};

template <typename StmtType>
struct Stmt: public IStmt {
	Stmt(std::shared_ptr<StmtType> stmt):
		IStmt(stmt->t),
		stmt(stmt) {}

	Pos begin() const {
		return stmt->begin();
	}

	Pos end() const {
		return stmt->end();
	}

	operator std::string() const {
		return std::string(*stmt);
	}

	std::shared_ptr<StmtType> stmt;
};

struct MacroArg: public Node {
	MacroArg(const Token& name, std::shared_ptr<String> default_value):
		Node(name), default_value(default_value) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (default_value) {
			return default_value->end();
		} else {
			return t.end();
		}
	}

	operator std::string() const {
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

struct MacroAction: public Node {
	MacroAction(const Token& macro_action,
		const Token& name,
		const std::vector<std::shared_ptr<MacroArg>>& args,
		std::shared_ptr<Action<ActionBlock>> action_block):
			Node(macro_action), name(name), args(args),
			action_block(action_block) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return action_block->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + name.value() + "(";
		for (auto arg: args) {
			result += std::string(*arg) + " ,";
		}
		result += ") ";
		result += std::string(*action_block);
		return result;
	}

	Token name;
	std::vector<std::shared_ptr<MacroArg>> args;
	std::shared_ptr<Action<ActionBlock>> action_block;
};

struct MacroActionCall: public Node {
	MacroActionCall(const Token& macro_name, const std::vector<std::shared_ptr<String>>& args):
		Node(macro_name), args(args) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return t.end();
	}

	operator std::string() const {
		std::string result = t.value() + ("(");
		for (auto arg: args) {
			result += std::string(*arg) + " ,";
		}
		result += ")";
		return result;
	}

	Token name() const {
		return t;
	}

	std::shared_ptr<MacroAction> macro_action;
	std::vector<std::shared_ptr<String>> args;
};


struct IAttrValue: public Node {
	using Node::Node;
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

	std::shared_ptr<AttrType> attr_value;
};

struct SimpleAttr: public Node {
	SimpleAttr(const Token& value):
		Node(value) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return t.end();
	}

	operator std::string() const {
		return t.value();
	}
};

struct BinaryAttr: public Node {
	BinaryAttr(const Token& _value):
		Node(Token(Token::category::binary, _value.value(), _value.begin(), _value.end())),
		value(_value) {}

	Pos begin() const {
		return value.begin();
	}

	Pos end() const {
		return value.end();
	}

	operator std::string() const {
		return value.value();
	}

	Token value;
};

struct StringAttr: public Node {
	StringAttr(std::shared_ptr<String> value):
		Node(value->t),
		value(value) {}

	Pos begin() const {
		return value->begin();
	}

	Pos end() const {
		return value->end();
	}

	operator std::string() const {
		return std::string(*value);
	}

	std::shared_ptr<String> value;
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

	Token open_brace;
	Token close_brace;
	std::vector<std::shared_ptr<Attr>> attrs;
};

struct Test: public Node {
	Test(std::shared_ptr<AttrBlock> attrs,
		const Token& test, const Token& name,
		const std::vector<Token>& parents_tokens,
		std::shared_ptr<CmdBlock> cmd_block):
		Node(test),
		attrs(attrs),
		name(name),
		parents_tokens(parents_tokens),
		cmd_block(cmd_block) {}

	Pos begin() const {
		if (attrs) {
			return attrs->begin();
		} else {
			return t.begin();
		}
	}

	Pos end() const {
		return cmd_block->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + name.value();
		return result; //for now
	}

	std::shared_ptr<AttrBlock> attrs;
	Token name;
	std::vector<Token> parents_tokens;
	std::list<std::shared_ptr<AST::Test>> parents;
	std::shared_ptr<CmdBlock> cmd_block;
	bool snapshots_needed = true;
	bool is_initialized = false;
	std::string description;
	std::chrono::system_clock::time_point start_timestamp;
	std::chrono::system_clock::time_point stop_timestamp;
};

struct Controller: public Node {
	Controller(const Token& controller, const Token& name, std::shared_ptr<AttrBlock> attr_block):
		Node(controller),
		name(name),
		attr_block(attr_block) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return attr_block->end();
	}

	operator std::string() const {
		return t.value() + " " + name.value() + " " + std::string(*attr_block);
	}

	Token name;
	std::shared_ptr<AttrBlock> attr_block;
};

struct Param: public Node {
	Param(const Token& param_token, const Token& name, std::shared_ptr<String> value):
		Node(param_token), name(name), value(value) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return value->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + name.value() + " " + std::string(*value);
		return result;
	}

	Token name;
	std::shared_ptr<String> value;
};

struct Program: public Node {
	Program (const std::vector<std::shared_ptr<IStmt>> stmts):
		Node(Token(Token::category::program, "program", Pos(),Pos())),
		stmts(stmts) {}

	Pos begin() const {
		return stmts[0]->begin();
	}

	Pos end() const {
		return stmts[stmts.size() - 1]->end();
	}

	operator std::string() const {
		std::string result;

		for (auto stmt: stmts) {
			result += std::string(*stmt);

		}

		return result;
	}

	std::vector<std::shared_ptr<IStmt>> stmts;
};

//here go expressions and everything with them

struct IFactor: public Node {
	using Node::Node;
};

//String, comparison, check or expr
template <typename FactorType>
struct Factor: public IFactor {
	Factor(const Token& not_token, std::shared_ptr<FactorType> factor):
		IFactor(factor->t),
		not_token(not_token),
		factor(factor) {}

	Pos begin() const {
		return factor->begin();
	}

	Pos end() const {
		return factor->end();
	}

	operator std::string() const {
		std::string result = "FACTOR: ";
		if (is_negated()) {
			result += "NOT ";
		}
		result += std::string(*factor);
		return result;
	}

	bool is_negated() const {
		return not_token.type() == Token::category::NOT;
	}

	Token not_token;
	std::shared_ptr<FactorType> factor;
};

struct Comparison: public Node {
	Comparison(const Token& op, std::shared_ptr<String> left, std::shared_ptr<String> right):
		Node(op), left(left), right(right) {}

	Pos begin() const {
		return left->begin();
	}

	Pos end() const {
		return right->end();
	}

	operator std::string() const {
		return std::string(*left) + " " + t.value() + " " + std::string(*right);
	}

	Token op() const {
		return t;
	}

	std::shared_ptr<String> left;
	std::shared_ptr<String> right;
};

struct Check: public Node {
	Check(const Token& check, std::shared_ptr<ISelectExpr> select_expr, std::shared_ptr<StringTokenUnion> timeout, std::shared_ptr<StringTokenUnion> interval):
		Node(check), select_expr(select_expr), timeout(timeout), interval(interval) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (interval) {
			return interval->end();
		} else if (timeout) {
			return timeout->end();
		} else {
			return select_expr->end();
		}
	}

	operator std::string() const {
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

	std::shared_ptr<ISelectExpr> select_expr;

	std::shared_ptr<StringTokenUnion> timeout;
	std::shared_ptr<StringTokenUnion> interval;
};

struct IExpr: public Node {
	using Node::Node;
};


//BinOp(AND, OR) or Factor
template <typename ExprType>
struct Expr: IExpr {
	Expr(std::shared_ptr<ExprType> expr):
		IExpr(expr->t),
		expr(expr) {}

	Pos begin() const {
		return expr->begin();
	}

	Pos end() const {
		return expr->end();
	}

	operator std::string() const {
		return std::string(*expr);
	}

	std::shared_ptr<ExprType> expr;
};


struct BinOp: public Node {
	BinOp(const Token& op, std::shared_ptr<IExpr> left, std::shared_ptr<IExpr> right):
		Node(op), left(left), right(right) {}

	Pos begin() const {
		return left->begin();
	}

	Pos end() const {
		return right->end();
	}

	operator std::string() const {
		return std::string("BINOP: ") + std::string(*left) + " " + t.value() + " " + std::string(*right);
	}

	Token op() const {
		return t;
	}

	std::shared_ptr<IExpr> left;
	std::shared_ptr<IExpr> right;
};

struct IfClause: public Node {
	IfClause(const Token& if_token, const Token& open_paren, std::shared_ptr<IExpr> expr,
		const Token& close_paren, std::shared_ptr<IAction> if_action, const Token& else_token,
		std::shared_ptr<IAction> else_action):
		Node(if_token),
		open_paren(open_paren),
		expr(expr),
		close_paren(close_paren),
		if_action(if_action),
		else_token(else_token),
		else_action(else_action)
	{}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (has_else()) {
			return else_action->end();
		} else {
			return if_action->end();
		}
	}

	operator std::string() const {
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
	std::shared_ptr<IExpr> expr;
	Token close_paren;
	std::shared_ptr<IAction> if_action;
	Token else_token;
	std::shared_ptr<IAction> else_action;
};

struct ICounterList: public Node {
	using Node::Node;
	virtual std::vector<std::string> values() const = 0;
};

template <typename CounterListType>
struct CounterList: public ICounterList {
	CounterList(std::shared_ptr<CounterListType> counter_list):
		ICounterList(counter_list->t),
		counter_list(counter_list) {}

	Pos begin() const {
		return counter_list->begin();
	}

	Pos end() const {
		return counter_list->end();
	}

	operator std::string() const {
		return std::string(*counter_list);
	}

	std::vector<std::string> values() const {
		return counter_list->values();
	}

	std::shared_ptr<CounterListType> counter_list;
};

struct Range: public Node {
	Range(const Token& range, std::shared_ptr<StringTokenUnion> r1, std::shared_ptr<StringTokenUnion> r2):
		Node(range), r1(r1), r2(r2) {}

	Pos begin() const {
		return r1->begin();
	}

	Pos end() const {
		if (r2) {
			return r2->end();
		} else {
			return r1->end();
		}
	}

	operator std::string() const {
		std::string result = t.value() + " " + std::string(*r1);
		if (r2) {
			result += " " + std::string(*r2);
		}
		return result;
	}

	std::vector<std::string> values() const {
		std::vector<std::string> result;

		uint32_t start = 0;
		uint32_t finish = 0;
		if (r2) {
			start = r1_num;
			finish = r2_num;
		} else {
			start = 0;
			finish = r1_num;
		}

		for (uint32_t i = start; i < finish; ++i) {
			result.push_back(std::to_string(i));
		}

		return result;
	}

	std::shared_ptr<StringTokenUnion> r1 = nullptr;
	std::shared_ptr<StringTokenUnion> r2 = nullptr;

	uint32_t r1_num = 0;
	uint32_t r2_num = 0;
};

struct ForClause: public Node {
	ForClause(const Token& for_token, const Token& counter,	std::shared_ptr<ICounterList> counter_list,
		std::shared_ptr<IAction> cycle_body, const Token& else_token,
		std::shared_ptr<IAction> else_action):
		Node(for_token),
		counter(counter),
		counter_list(counter_list),
		cycle_body(cycle_body),
		else_token(else_token),
		else_action(else_action) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		if (else_token) {
			return else_action->end();
		} else {
			return cycle_body->end();
		}
	}

	operator std::string() const {
		std::string result = t.value() + "(" + counter.value() + " IN " + std::string(*counter_list) + ")";

		result += std::string(*cycle_body);
		if (else_action) {
			result += std::string(*else_action);
		}
		return result;
	}

	Token counter;
	std::shared_ptr<ICounterList> counter_list = nullptr;
	std::shared_ptr<IAction> cycle_body;

	Token else_token;
	std::shared_ptr<IAction> else_action;
};

struct CycleControl: public Node {
	CycleControl(const Token& control_token): Node(control_token) {}

	Pos begin() const {
		return t.begin();
	}

	Pos end() const {
		return t.end();
	}

	operator std::string() const {
		return t.value();
	}
};

}

