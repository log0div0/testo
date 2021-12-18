
#pragma once

#include "Base.hpp"

// !!!!!!!!!!!!!!!!!
// THIS FILE IS ONLY FOR INTERNAL IR USE ONLY
// !!!!!!!!!!!!!!!!!

namespace IR {

struct OptionSeq : Node<AST::OptionSeq> {
	using Node::Node;

	template <typename T>
	T get(const std::string& name, const std::string& default_param) const {
		std::shared_ptr<typename T::ASTType> p = nullptr;
		if (auto option = ast_node->get(name)) {
			p = std::dynamic_pointer_cast<typename T::ASTType>(option->value);
			if (!p) {
				throw std::runtime_error("Failed to cast \"" + name + "\" option to the target type");
			}
		} else {
			p = IR::program->get_top_level_param_ast<typename T::ASTType>(default_param);
		}
		return {p, stack};
	}

	template <typename T>
	T get(const std::string& name) const {
		std::shared_ptr<typename T::ASTType> p = nullptr;
		if (auto option = ast_node->get(name)) {
			p = std::dynamic_pointer_cast<typename T::ASTType>(option->value);
			if (!p) {
				throw std::runtime_error("Failed to cast \"" + name + "\" option to the target type");
			}
			return {p, stack};
		} else {
			throw std::runtime_error("Option \"" + name + "\" is not specified");
		}
	}

	bool has(const std::string& name) {
		return ast_node->get(name) != nullptr;
	}
};

}
