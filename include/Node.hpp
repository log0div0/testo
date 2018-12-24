
#pragma once

#include <Pos.hpp>
#include <Token.hpp>
#include <vector>
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
		for (size_t i = 1; i < buttons.size(); i++) {
			result += "+" + buttons[i].value();
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
		return std::string(*action) + delim.value();
	}

	void set_delim (const Token& delim) {
		this->delim = delim;
	}

	std::shared_ptr<ActionType> action;
	Token delim;
};

struct Type: public Node {
	Type(const Token& type, const Token& text_token):
		Node(type), text_token(text_token) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return text_token.pos();
	}

	operator std::string() const {
		return t.value() + " " + text_token.value();
	}

	Token text_token;
	std::string text() const {
		return text_token.value().substr(1, text_token.value().length() - 2);
	}
};

struct Wait: public Node {
	Wait(const Token& wait, const Token& text_token, Token for_, Token time_interval):
		Node(wait), text_token(text_token), time_interval(time_interval), for_(for_) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		if (time_interval) {
			return time_interval.pos();
		} else {
			return text_token.pos();
		}
	}

	operator std::string() const {
		std::string result = t.value();

		if (text_token) {
			result += " " + text_token.value();
		}

		if (time_interval) {
			result += " " + for_.value() + " " + time_interval.value();
		} 
		return result;
	}

	std::string text() const {
		if (text_token) {
			return text_token.value().substr(1, text_token.value().length() - 2);
		} else {
			return "";
		}
	}

	Token text_token;
	Token time_interval;
	Token for_;
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

//Also is used for unplug
struct Plug: public Node {
	Plug(const Token& plug, const Token& type, const Token& name):
		Node(plug),
		type(type),
		name_token(name) {}

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

	std::string name() const {
		if (type.type() == Token::category::dvd) {
			return name_token.value().substr(1, name_token.value().length() - 2);
		} else {
			return name_token.value();
		}
	}

	Token type; //nic or flash
	Token name_token; //name of resource to be plugged/unplugged
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

struct Exec: public Node {
	Exec(const Token& exec, const Token& process, const Token& commands):
		Node(exec),
		process_token(process),
		commands_token(commands) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return commands_token.pos();
	}

	operator std::string() const {
		return t.value() + " " + process_token.value() + " " + commands_token.value();
	}

	std::string script() const {
		if (commands_token.type() == Token::category::dbl_quoted_string) {
			return commands_token.value().substr(1, commands_token.value().length() - 2);
		} else if (commands_token.type() == Token::category::multiline_string) {
			return commands_token.value().substr(3, commands_token.value().length() - 6);
		} else {
			throw std::runtime_error(std::string(begin()) + ": making script error: unsupported commands token: " + commands_token.value());
		}
	}

	Token process_token;
	Token commands_token;
};

struct Assignment: public Node {
	Assignment(const Token& left, const Token& assign, const Token& right):
		Node(assign),
		left(left),
		right(right) {}

	Pos begin() const {
		return left.pos();
	}

	Pos end() const {
		return right.pos();
	}

	operator std::string() const {
		return left.value() + t.value() + right.value();
	}

	std::string value() const {
		if (right.type() == Token::category::dbl_quoted_string) {
			return right.value().substr(1, right.value().length() - 2);
		} else {
			throw std::runtime_error(std::string(right.pos()) + ": error: unknown token type");
		}
	}

	Token left;
	Token right;
};

struct Set: public Node {
	Set(const Token& set, const std::vector<std::shared_ptr<Assignment>>& assignments):
		Node(set),
		assignments(assignments) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return assignments[assignments.size() - 1]->end();
	}

	operator std::string() const {
		std::string result = t.value();
		for (auto assignment: assignments) {
			result += " ";
			result += *assignment;
		}
		return result;
	}

	std::vector<std::shared_ptr<Assignment>> assignments;
};

struct CopyTo: public Node {
	CopyTo(const Token& copyto, const Token& from, const Token& to):
		Node(copyto),
		from_token(from),
		to_token(to) {}

	Pos begin() const {
		return from_token.pos();
	}

	Pos end() const {
		return to_token.pos();
	}

	operator std::string() const {
		return t.value() + " " + from_token.value() + " " + to_token.value();
	}

	std::string from() const {
		return from_token.value().substr(1, from_token.value().length() - 2);
	}

	std::string to() const {
		return to_token.value().substr(1, to_token.value().length() - 2);
	}

	Token from_token;
	Token to_token;
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
//may be machine declaration, snapshot declaration and test
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

struct Snapshot: public Node {
	Snapshot(const Token& snapshot, const Token& name,
		const Token& parent_name, std::shared_ptr<Action<ActionBlock>> action_block):
		Node(snapshot),
		name(name),
		parent_name(parent_name),
		action_block(action_block),
		parent() {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return action_block->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + name.value();
		if (parent_name) {
			result += ":(" + parent_name.value() + ")";
		}
		result += " " + std::string(*action_block);
		return result;
	}

	Token name;
	Token parent_name;
	std::shared_ptr<Action<ActionBlock>> action_block;
	std::shared_ptr<Snapshot> parent;
};

struct VmState: public Node {
	VmState(const Token& name, const Token& snapshot_name):
		Node(Token(Token::category::vm_state, "", Pos())),
		name(name),
		snapshot_name(snapshot_name),
		snapshot() {}

	Pos begin() const {
		return name.pos();
	}

	Pos end() const {
		if (snapshot) {
			return snapshot_name.pos();
		} else {
			return name.pos();
		}
	}

	operator std::string() const {
		std::string result = name.value();
		if (snapshot) {
			result += "(" + snapshot_name.value() + ")";
		}
		return result;
	}

	Token name;
	Token snapshot_name;
	std::shared_ptr<Snapshot> snapshot;
};

struct Test: public Node {
	Test(const Token& test, const Token& name,
		const std::vector<std::shared_ptr<VmState>>& vms,
		std::shared_ptr<CmdBlock> cmd_block):
		Node(test),
		name(name),
		vms(vms),
		cmd_block(cmd_block) {}

	Pos begin() const {
		return t.pos();
	}

	Pos end() const {
		return cmd_block->end();
	}

	operator std::string() const {
		std::string result = t.value() + " " + name.value();
		return result; //for now
	}

	Token name;
	std::vector<std::shared_ptr<VmState>> vms;
	std::shared_ptr<CmdBlock> cmd_block;
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

}

