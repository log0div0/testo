
#pragma once

#include "Object.hpp"
#include "../Exceptions.hpp"
#include <fmt/format.h>

namespace IR {

template <typename Item>
std::shared_ptr<AST::Block<Item>> parse_block(const std::vector<Token>& tokens);

template <>
std::shared_ptr<AST::Block<AST::Stmt>> parse_block(const std::vector<Token>& tokens);

template <>
std::shared_ptr<AST::Block<AST::Cmd>> parse_block(const std::vector<Token>& tokens);

template <>
std::shared_ptr<AST::Block<AST::Action>> parse_block(const std::vector<Token>& tokens);

struct Macro: Object<AST::Macro> {
	static std::string type_name() { return "macro"; }

	template <typename Item>
	std::shared_ptr<AST::Block<Item>> get_body() {
		if (!ast_node->block) {
			ast_node->block = parse_block<Item>(ast_node->body_tokens);
		}
		return std::dynamic_pointer_cast<AST::Block<Item>>(ast_node->block);
	}

	void validate();
};

struct MacroCall: Node<AST::IMacroCall> {
	using Node<AST::IMacroCall>::Node;

	const std::shared_ptr<IR::Macro> get_macro() const;
	std::vector<std::pair<std::string, std::string>> args() const;
	std::map<std::string, std::string> vars() const;

	template <typename Item, typename Visitor>
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
			throw std::runtime_error(fmt::format("{}: Error: expected at least {} args, {} provided", std::string(ast_node->name.begin()),
				macro->ast_node->args.size() - args_with_default, ast_node->args.size()));
		}

		if (ast_node->args.size() > macro->ast_node->args.size()) {
			throw std::runtime_error(fmt::format("{}: Error: expected at most {} args, {} provided", std::string(ast_node->name.begin()),
				macro->ast_node->args.size(), ast_node->args.size()));
		}

		StackPusher<Visitor> new_ctx(visitor, macro->new_stack(vars()));

		std::shared_ptr<AST::Block<Item>> p = macro->get_body<Item>();
		if (p == nullptr) {
			throw std::runtime_error(std::string(ast_node->name.begin()) + ": Error: the \"" + ast_node->name.value() + "\" macro does not contain " + Item::desc() + ", as expected");
		}

		try {
			visitor->visit_macro_body(p);
		} catch (const std::exception& error) {
			std::throw_with_nested(MacroException(ast_node));
		}
	}

	template <typename Item, typename Visitor>
	void visit_interpreter(Visitor* visitor) const {
		const std::shared_ptr<IR::Macro> macro = get_macro();

		StackPusher<Visitor> new_ctx(visitor, macro->new_stack(vars()));

		std::shared_ptr<AST::Block<Item>> p = macro->get_body<Item>();
		if (p == nullptr) {
			throw std::runtime_error("Should never happen");
		}

		try {
			visitor->visit_macro_body(p);
		} catch (const std::exception& error) {
			std::throw_with_nested(MacroException(ast_node));
		}
	}
};

}
