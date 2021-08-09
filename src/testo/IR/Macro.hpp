
#pragma once

#include "Object.hpp"

namespace IR {

template <typename BlockType>
std::shared_ptr<BlockType> parse_block(const std::vector<Token>& tokens);

template <>
std::shared_ptr<AST::StmtBlock> parse_block(const std::vector<Token>& tokens);

template <>
std::shared_ptr<AST::CmdBlock> parse_block(const std::vector<Token>& tokens);

template <>
std::shared_ptr<AST::ActionBlock> parse_block(const std::vector<Token>& tokens);

struct Macro: Object<AST::Macro> {
	static std::string type_name() { return "macro"; }

	template <typename MacroBodyType>
	std::shared_ptr<AST::MacroBody<MacroBodyType>> get_body() {
		if (!ast_node->body) {
			auto block = parse_block<typename MacroBodyType::BlockType>(ast_node->body_tokens);
			auto body = std::make_shared<MacroBodyType>(block);
			ast_node->body = std::make_shared<AST::MacroBody<MacroBodyType>>(body);
		}
		return std::dynamic_pointer_cast<AST::MacroBody<MacroBodyType>>(ast_node->body);
	}

	void validate();
};

struct MacroCall: Node<AST::IMacroCall> {
	using Node<AST::IMacroCall>::Node;

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
