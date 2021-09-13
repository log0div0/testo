#pragma once

#include "Base.hpp"

namespace IR {

template <typename ASTType>
struct Command: Node<ASTType> {
	using Node<ASTType>::Node;

	std::string entity() const;
};

struct RegularCommand: Command<AST::RegularCmd> {
	using Command<AST::RegularCmd>::Command;
	std::string entity() const;
};

}