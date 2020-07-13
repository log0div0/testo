
#pragma once

#include "../AST.hpp"
#include "../Stack.hpp"

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

struct Press: Action<AST::Press> {
	using Action<AST::Press>::Action;
	std::string interval() const;
};

struct Type: Action<AST::Type> {
	using Action<AST::Type>::Action;
	std::string text() const;
	std::string interval() const;
};

struct Wait: Action<AST::Wait> {
	using Action<AST::Wait>::Action;
	std::string timeout() const;
	std::string interval() const;
};

struct MouseSelectable: Action<AST::MouseSelectable> {
	using Action<AST::MouseSelectable>::Action;
	std::string timeout() const;
};

struct Exec: Action<AST::Exec> {
	using Action<AST::Exec>::Action;
	std::string timeout() const;
	std::string text() const;
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

}
