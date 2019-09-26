
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
struct Word: public Node {
	Word(const std::vector<Token> parts):
		Node(Token(Token::category::word, "word", parts[0].pos())),
		parts(parts) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return parts[parts.size() - 1].pos();
	}

	operator std::string() const {
		std::string result;
		for (auto& part: parts) {
			result += part.value();
		}
		return t.value();
	}

	std::vector<Token> parts;
};

struct KeySpec: public Node {
	KeySpec(const std::vector<Token>& buttons, const Token& times):
		Node(Token(Token::category::key_spec, "key_spec", Pos())),
		buttons(buttons),
		times(times) {}

	Pos begin() const {
		return buttons[0].pos();
	}

	Pos end() const {
		if (times) {
			return times.pos();
		} else {
			return buttons[buttons.size() - 1].pos();
		}
	}

	operator std::string() const {
		std::string result = buttons[0].value();
		for (size_t i = 1; i < buttons.size(); i++) {
			result += "+" + buttons[i].value();
		}
		if (times) {
			result += "*" + times.value();
		}
		return result;
	}

	std::vector<std::string> get_buttons() const {
		std::vector<std::string> result;

		for (auto& button: buttons) {
			result.push_back(button.value());
		}

		return result;
	}

	std::string get_buttons_str() const {
		std::string result = buttons[0].value();
		std::transform(result.begin(), result.end(), result.begin(), ::toupper);
		for (size_t i = 1; i < buttons.size(); i++) {
			auto button_str = buttons[i].value();
			std::transform(button_str.begin(), button_str.end(), button_str.begin(), ::toupper);
			result += "+" + button_str;
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

	std::vector<Token> buttons;
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
		return delim.pos();
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
		return t.pos();
	}

	Pos end() const {
		return t.pos();
	}

	operator std::string() const {
		return "";
	}
};

struct Abort: public Node {
	Abort(const Token& abort, std::shared_ptr<Word> message):
		Node(abort), message(message) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return message->end();
	}

	operator std::string() const {
		return t.value() + " " + std::string(*message);
	}

	std::shared_ptr<Word> message;
};

struct Print: public Node {
	Print(const Token& print, std::shared_ptr<Word> message):
		Node(print), message(message) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return message->end();
	}

	operator std::string() const {
		return t.value() + " " + std::string(*message);
	}

	std::shared_ptr<Word> message;
};

struct Type: public Node {
	Type(const Token& type, std::shared_ptr<Word> text_word):
		Node(type), text_word(text_word) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return text_word->end();
	}

	operator std::string() const {
		return t.value() + " " + std::string(*text_word);
	}

	std::shared_ptr<Word> text_word;
};

struct Assignment: public Node {
	Assignment(const Token& left, const Token& assign, std::shared_ptr<Word> right):
		Node(assign),
		left(left),
		right(right) {}

	Pos begin() const {
		return left.pos();
	}

	Pos end() const {
		return right->end();
	}

	operator std::string() const {
		return left.value() + t.value() + std::string(*right);
	}

	Token left;
	std::shared_ptr<Word> right;
};

struct Wait: public Node {
	Wait(const Token& wait, std::shared_ptr<Word> text_word,
	const std::vector<std::shared_ptr<Assignment>>& params, const Token& timeout, const Token& time_interval):
		Node(wait), text_word(text_word), params(params), timeout(timeout), time_interval(time_interval) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		if (timeout) {
			return timeout.pos();
		} else {
			return text_word->end();
		}
	}

	operator std::string() const {
		std::string result = t.value();

		if (text_word) {
			result += " " + std::string(*text_word);
		}

		if (timeout) {
			result += " " + timeout.value() + " " + time_interval.value();
		}
		return result;
	}

	std::shared_ptr<Word> text_word;
	std::vector<std::shared_ptr<Assignment>> params;
	Token timeout;
	Token time_interval;
};

struct Press: public Node {
	Press(const Token& press, const std::vector<std::shared_ptr<KeySpec>> keys):
		Node(press), keys(keys) {}

	Pos begin() const {
		return keys[0]->begin();
	}

	Pos end() const {
		return keys[keys.size() - 1]->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + std::string(*keys[0]);

		for (size_t i = 1; i < keys.size(); i++) {
			result += ", " + std::string(*keys[i]);
		}

		return result;
	}

	std::vector<std::shared_ptr<KeySpec>> keys;
};

struct MouseMove: public Node {
	MouseMove(const Token& mouse, const Token& move, const Token& dx, const Token& dy):
		Node(mouse), move(move), dx_token(dx), dy_token(dy) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return dy_token.pos();
	}

	operator std::string() const {
		std::string result = t.value() + " " + move.value() + " " + dx_token.value() + " " + dy_token.value();

		return result;
	}

	int dx() {
		std::stoi(dx_token.value());
	}

	int dy() {
		std::stoi(dy_token.value());
	}

	Token move, dx_token, dy_token;
};

struct MouseClick: public Node {
	MouseClick(const Token& mouse, const Token& click_type):
		Node(mouse), click_type(click_type) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return click_type.pos();
	}

	operator std::string() const {
		std::string result = t.value() + " " + click_type.value();

		return result;
	}

	Token click_type;
};

//Also is used for unplug
struct Plug: public Node {
	Plug(const Token& plug, const Token& type, const Token& name, std::shared_ptr<Word> path):
		Node(plug),
		type(type),
		name_token(name),
		path(path) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return name_token.pos();
	}

	operator std::string() const {
		return t.value() + " " + type.value() + " " + name_token.value();
	}

	bool is_on() const {
		return (t.type() == Token::category::plug);
	}

	Token type; //nic or flash or dvd
	Token name_token; //name of resource to be plugged/unplugged
	std::shared_ptr<Word> path; //used only for dvd
};


struct Start: public Node {
	Start(const Token& start):
		Node(start) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return t.pos();
	}

	operator std::string() const {
		return t.value();
	}
};

struct Stop: public Node {
	Stop(const Token& stop):
		Node(stop) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return t.pos();
	}

	operator std::string() const {
		return t.value();
	}
};

struct Shutdown: public Node {
	Shutdown(const Token& shutdown, const Token& timeout, const Token& time_interval):
		Node(shutdown), timeout(timeout), time_interval(time_interval) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		if (timeout) {
			return time_interval.pos();
		}
		return t.pos();
	}

	operator std::string() const {
		std::string result = t.value();
		if (timeout) {
			result += " timeout " + time_interval.value();
		}
		return result;
	}

	Token timeout;
	Token time_interval;
};

struct Exec: public Node {
	Exec(const Token& exec, const Token& process, std::shared_ptr<Word> commands, const Token& timeout, const Token& time_interval):
		Node(exec),
		process_token(process),
		commands(commands), timeout(timeout), time_interval(time_interval) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		if (timeout) {
			return time_interval.pos();
		}
		return commands->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + process_token.value() + " " + std::string(*commands);
		if (timeout) {
			result += " timeout " + time_interval.value();
		}
		return result;
	}

	Token process_token;
	std::shared_ptr<Word> commands;
	Token timeout;
	Token time_interval;
};

//Now this node holds actions copyto and copyfrom
//Cause they're really similar
struct Copy: public Node {
	Copy(const Token& copy, std::shared_ptr<Word> from, std::shared_ptr<Word> to, const Token& timeout, const Token& time_interval):
		Node(copy),
		from(from),
		to(to), timeout(timeout), time_interval(time_interval) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		if (timeout) {
			return time_interval.pos();
		}
		return to->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + std::string(*from) + " " + std::string(*to);
		if (timeout) {
			result += " timeout " + time_interval.value();
		}
		return result;
	}

	//return true if we copy to guest,
	//false if from guest to host
	bool is_to_guest() const {
		return t.type() == Token::category::copyto;
	}

	std::shared_ptr<Word> from;
	std::shared_ptr<Word> to;
	Token timeout;
	Token time_interval;
};

struct ActionBlock: public Node {
	ActionBlock(const Token& open_brace, const Token& close_brace, std::vector<std::shared_ptr<IAction>> actions):
		Node(Token(Token::category::action_block, "action_block", Pos())),
		open_brace(open_brace),
		close_brace(close_brace),
		actions(actions) {}

	Pos begin() const {
		return open_brace.pos();
	}

	Pos end() const {
		return close_brace.pos();
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
		Node(Token(Token::category::cmd, "cmd", Pos())),
		vms(vms),
		action(action) {}

	Pos begin() const {
		return vms.begin()->pos();
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
		Node(Token(Token::category::cmd_block, "cmd_block", Pos())),
		open_brace(open_brace),
		close_brace(close_brace),
		commands(commands) {}

	Pos begin() const {
		return open_brace.pos();
	}

	Pos end() const {
		return close_brace.pos();
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

struct Macro: public Node {
	Macro(const Token& macro,
		const Token& name,
		const std::vector<Token>& params,
		std::shared_ptr<Action<ActionBlock>> action_block):
			Node(macro), name(name), params(params),
			action_block(action_block) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return action_block->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + name.value() + "(";
		for (auto param: params) {
			result += param.value() + " ,";
		}
		result += ") ";
		result += std::string(*action_block);
		return result;
	}

	Token name;
	std::vector<Token> params;
	std::shared_ptr<Action<ActionBlock>> action_block;
};

struct MacroCall: public Node {
	MacroCall(const Token& macro_name, const std::vector<std::shared_ptr<Word>>& params):
		Node(macro_name), params(params) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return t.pos();
	}

	operator std::string() const {
		std::string result = t.value() + ("(");
		for (auto param: params) {
			result += std::string(*param) + " ,";
		}
		result += ")";
		return result;
	}

	Token name() const {
		return t;
	}

	std::shared_ptr<Macro> macro;
	std::vector<std::shared_ptr<Word>> params;
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
		return t.pos();
	}

	Pos end() const {
		return t.pos();
	}

	operator std::string() const {
		return t.value();
	}
};

struct BinaryAttr: public Node {
	BinaryAttr(const Token& _value):
		Node(Token(Token::category::binary, _value.value(), _value.pos())),
		value(_value) {}

	Pos begin() const {
		return value.pos();
	}

	Pos end() const {
		return value.pos();
	}

	operator std::string() const {
		return value.value();
	}

	Token value;
};

struct WordAttr: public Node {
	WordAttr(std::shared_ptr<Word> value):
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

	std::shared_ptr<Word> value;
};

struct Attr: public Node {
	Attr(const Token& name, const Token& id, std::shared_ptr<IAttrValue> value):
		Node(Token(Token::category::attr, "", Pos())),
		name(name),
		id(id),
		value(value) {}

	Pos begin() const {
		return name.pos();
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
		Node(Token(Token::category::attr_block, "", Pos())),
		open_brace(open_brace),
		close_brace(close_brace),
		attrs(attrs) {}

	Pos begin() const {
		return open_brace.pos();
	}

	Pos end() const {
		return close_brace.pos();
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
			return t.pos();
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
		return t.pos();
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

struct Program: public Node {
	Program (const std::vector<std::shared_ptr<IStmt>> stmts):
		Node(Token(Token::category::program, "program", Pos())),
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

//Word, comparison, check or expr
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
	Comparison(const Token& op, std::shared_ptr<Word> left, std::shared_ptr<Word> right):
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

	std::shared_ptr<Word> left;
	std::shared_ptr<Word> right;
};

struct Check: public Node {
	Check(const Token& check, std::shared_ptr<Word> text_word,
	const std::vector<std::shared_ptr<Assignment>>& params):
		Node(check), text_word(text_word), params(params) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		if (params.size()) {
			return params[params.size() - 1]->end();
		} else {
			return text_word->end();
		}
	}

	operator std::string() const {
		std::string result = t.value();

		if (text_word) {
			result += " " + std::string(*text_word);
		}

		return result;
	}

	std::shared_ptr<Word> text_word;
	std::vector<std::shared_ptr<Assignment>> params;
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
		return t.pos();
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

struct ForClause: public Node {
	ForClause(const Token& for_token, const Token& counter, const Token& in, const Token& start,
		const Token& double_dot, const Token& finish, std::shared_ptr<IAction> cycle_body):
		Node(for_token),
		counter(counter),
		in(in),
		start_(start),
		double_dot(double_dot),
		finish_(finish),
		cycle_body(cycle_body) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return cycle_body->end();
	}

	operator std::string() const {
		return t.value();
	}

	uint32_t start() const {
		return std::stoul(start_.value());
	}

	uint32_t finish() const {
		return std::stoul(finish_.value());
	}

	Token counter, in, start_, double_dot, finish_;
	std::shared_ptr<IAction> cycle_body;
};

struct CycleControl: public Node {
	CycleControl(const Token& control_token): Node(control_token) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return t.pos();
	}

	operator std::string() const {
		return t.value();
	}
};

}

