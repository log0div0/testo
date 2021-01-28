
#pragma once

#include "../AST.hpp"
#include "../Stack.hpp"
#include "../TemplateLiterals.hpp"
#include "../Exceptions.hpp"
#include "Macro.hpp"
#include <fmt/format.h>

namespace IR {

template <typename ASTType>
struct Action {
	Action(std::shared_ptr<ASTType> ast_node_, std::shared_ptr<StackNode> stack_):
		ast_node(std::move(ast_node_)), stack(std::move(stack_)) {}
	std::shared_ptr<ASTType> ast_node;
	std::shared_ptr<StackNode> stack;
};

struct Abort: Action<AST::Abort> {
	using Action<AST::Abort>::Action;
	std::string message() const;
};

struct Print: Action<AST::Print> {
	using Action<AST::Print>::Action;
	std::string message() const;
};

struct Press: Action<AST::Press> {
	using Action<AST::Press>::Action;
	std::string interval() const;
};

struct KeySpec: Action<AST::KeySpec> {
	using Action<AST::KeySpec>::Action;
	int32_t times() const;
};

struct Hold: Action<AST::Hold> {
	using Action<AST::Hold>::Action;
	std::vector<std::string> buttons() const;
};

struct Release: Action<AST::Release> {
	using Action<AST::Release>::Action;
	std::vector<std::string> buttons() const;
};

struct Type: Action<AST::Type> {
	using Action<AST::Type>::Action;
	std::string text() const;
	std::string interval() const;
};

struct Wait: Action<AST::Wait> {
	using Action<AST::Wait>::Action;
	std::string select_expr() const;
	std::string timeout() const;
	std::string interval() const;
};

struct Sleep: Action<AST::Sleep> {
	using Action<AST::Sleep>::Action;
	std::string timeout() const;
};

struct Mouse: Action<AST::Mouse> {
	using Action<AST::Mouse>::Action;
};

struct MouseMoveClick: Action<AST::MouseMoveClick> {
	using Action<AST::MouseMoveClick>::Action;

	std::string event_type() const;
};

struct MouseCoordinates: Action<AST::MouseCoordinates> {
	using Action<AST::MouseCoordinates>::Action;

	std::string x() const;
	std::string y() const;

	bool x_is_relative() const;
	bool y_is_relative() const;
};

struct MouseSelectable: Action<AST::MouseSelectable> {
	using Action<AST::MouseSelectable>::Action;
	std::string where_to_go() const;
	std::string timeout() const;
};

struct SelectJS: Action<AST::SelectJS> {
	using Action<AST::SelectJS>::Action;
	std::string script() const;
};

struct SelectImg: Action<AST::SelectImg> {
	using Action<AST::SelectImg>::Action;
	fs::path img_path() const;
};

struct SelectHomm3: Action<AST::SelectHomm3> {
	using Action<AST::SelectHomm3>::Action;
	std::string id() const;
};

struct SelectText: Action<AST::SelectText> {
	using Action<AST::SelectText>::Action;
	std::string text() const;
};

struct MouseHold: Action<AST::MouseHold> {
	using Action<AST::MouseHold>::Action;
	std::string button() const;
};

struct MouseRelease:Action<AST::MouseRelease> {
	using Action<AST::MouseRelease>::Action;
};

struct Plug: Action<AST::Plug> {
	using Action<AST::Plug>::Action;
	bool is_on() const;
};

struct PlugFlash: Action<AST::PlugFlash> {
	using Action<AST::PlugFlash>::Action;
	std::string name() const;
};

struct PlugNIC: Action<AST::PlugNIC> {
	using Action<AST::PlugNIC>::Action;
	std::string name() const;
};

struct PlugLink: Action<AST::PlugLink> {
	using Action<AST::PlugLink>::Action;
	std::string name() const;
};

struct PlugHostDev: Action<AST::PlugHostDev> {
	using Action<AST::PlugHostDev>::Action;
	std::string type() const;
	std::string addr() const;
};

struct PlugDVD: Action<AST::PlugDVD> {
	using Action<AST::PlugDVD>::Action;
	fs::path path() const;
};

struct Start:Action<AST::Start> {
	using Action<AST::Start>::Action;
};

struct Stop:Action<AST::Stop> {
	using Action<AST::Stop>::Action;
};

struct Shutdown:Action<AST::Shutdown> {
	using Action<AST::Shutdown>::Action;

	std::string timeout() const;
};

struct Exec: Action<AST::Exec> {
	using Action<AST::Exec>::Action;
	std::string interpreter() const;
	std::string timeout() const;
	std::string script() const;
};

struct Copy: Action<AST::Copy> {
	using Action<AST::Copy>::Action;
	std::string timeout() const;
	std::string from() const;
	std::string to() const;
};

struct Check: Action<AST::Check> {
	using Action<AST::Check>::Action;
	std::string timeout() const;
	std::string interval() const;
};

struct CycleControl: Action<AST::CycleControl> {
	using Action<AST::CycleControl>::Action;
	std::string type() const;
};

struct StringTokenUnion: Action<AST::StringTokenUnion> {
	using Action<AST::StringTokenUnion>::Action;
	std::string resolve() const;
};

struct MacroCall: Action<AST::MacroCall> {
	using Action<AST::MacroCall>::Action;

	const std::shared_ptr<IR::Macro> get_macro() const;
	std::vector<std::pair<std::string, std::string>> args() const;
	std::map<std::string, std::string> vars() const;

	template <typename MacroBodyType, typename Visitor>
	void visit_semantic(Visitor* visitor) const {
		const std::shared_ptr<IR::Macro> macro = get_macro();

		visitor->visit_macro(macro);

		uint32_t args_with_default = 0;

		for (auto arg: macro->ast_node->args) {
			if (arg->default_value) {
				args_with_default++;
			}
		}

		if (ast_node->args.size() < macro->ast_node->args.size() - args_with_default) {
			throw std::runtime_error(fmt::format("{}: Error: expected at least {} args, {} provided", std::string(ast_node->begin()),
				macro->ast_node->args.size() - args_with_default, ast_node->args.size()));
		}

		if (ast_node->args.size() > macro->ast_node->args.size()) {
			throw std::runtime_error(fmt::format("{}: Error: expected at most {} args, {} provided", std::string(ast_node->begin()),
				macro->ast_node->args.size(), ast_node->args.size()));
		}

		StackPusher<Visitor> new_ctx(visitor, macro->new_stack(vars()));

		std::shared_ptr<AST::MacroBody<MacroBodyType>> p = macro->get_body<MacroBodyType>();
		if (p == nullptr) {
			throw std::runtime_error(std::string(ast_node->begin()) + ": Error: the \"" + ast_node->name().value() + "\" macro does not contain " + MacroBodyType::desc() + ", as expected");
		}

		try {
			visitor->visit_macro_body(p->macro_body);
		} catch (const std::exception& error) {
			std::throw_with_nested(MacroException(ast_node));
		}
	}

	template <typename MacroBodyType, typename Visitor>
	void visit_interpreter(Visitor* visitor) const {
		const std::shared_ptr<IR::Macro> macro = get_macro();

		StackPusher<Visitor> new_ctx(visitor, macro->new_stack(vars()));

		std::shared_ptr<AST::MacroBody<MacroBodyType>> p = macro->get_body<MacroBodyType>();
		if (p == nullptr) {
			throw std::runtime_error("Should never happen");
		}

		try {
			visitor->visit_macro_body(p->macro_body);
		} catch (const std::exception& error) {
			std::throw_with_nested(MacroException(ast_node));
		}
	}
};

}
