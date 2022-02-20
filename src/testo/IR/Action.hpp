
#pragma once

#include "../Mouse.hpp"
#include "../Keyboard.hpp"
#include "Base.hpp"

namespace IR {

struct KeyCombination: MaybeUnparsed<AST::IKeyCombination, AST::KeyCombination> {
	using MaybeUnparsed::MaybeUnparsed;
	std::vector<KeyboardButton> buttons() const;
	std::string to_string() const;
};

struct Abort: Node<AST::Abort> {
	using Node<AST::Abort>::Node;
	std::string message() const;
};

struct Print: Node<AST::Print> {
	using Node<AST::Print>::Node;
	std::string message() const;
};

struct REPL: Node<AST::REPL> {
	using Node<AST::REPL>::Node;
};

struct Press: Node<AST::Press> {
	using Node<AST::Press>::Node;
	TimeInterval interval() const;
};

struct KeySpec: Node<AST::KeySpec> {
	using Node<AST::KeySpec>::Node;
	KeyCombination combination() const;
	int32_t times() const;
};

struct Hold: Node<AST::Hold> {
	using Node<AST::Hold>::Node;
	KeyCombination combination() const;
};

struct Release: Node<AST::Release> {
	using Node<AST::Release>::Node;
	KeyCombination combination() const;
};

struct Type: Node<AST::Type> {
	Type(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
		Node(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_)) {}

	String text() const;
	TimeInterval interval() const;
	KeyCombination autoswitch() const;
	bool use_autoswitch() const;
	void validate() const;

private:
	std::shared_ptr<VarMap> var_map;
};

struct Wait: Node<AST::Wait> {
	Wait(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
		Node(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_)) {}

	SelectExpr select_expr() const;
	TimeInterval timeout() const;
	TimeInterval interval() const;

private:
	std::shared_ptr<VarMap> var_map;
};

struct Sleep: Node<AST::Sleep> {
	using Node<AST::Sleep>::Node;
	TimeInterval timeout() const;
};

struct Mouse: Node<AST::Mouse> {
	using Node<AST::Mouse>::Node;
};

struct MouseMoveClick: Node<AST::MouseMoveClick> {
	using Node<AST::MouseMoveClick>::Node;

	std::string event_type() const;
};

struct MouseCoordinates: Node<AST::MouseCoordinates> {
	using Node<AST::MouseCoordinates>::Node;

	std::string x() const;
	std::string y() const;

	bool x_is_relative() const;
	bool y_is_relative() const;
};

struct MouseSelectable: Node<AST::MouseSelectable> {
	MouseSelectable(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
		Node(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_)) {}

	std::string to_string() const;
	TimeInterval timeout() const;

private:
	std::shared_ptr<VarMap> var_map;
};

struct SelectJS: Node<AST::SelectJS> {
	SelectJS(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
		Node(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_)) {}

	std::string script() const;

private:
	std::shared_ptr<VarMap> var_map;
};

struct File: String {
	using String::String;

	fs::path path() const;
	std::string signature() const;
	void validate() const;
};

struct SelectImg: Node<AST::SelectImg> {
	SelectImg(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
		Node(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_)) {}

	File img() const;

private:
	std::shared_ptr<VarMap> var_map;
};

struct SelectText: Node<AST::SelectText> {
	SelectText(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
		Node(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_)) {}

	std::string text() const;

private:
	std::shared_ptr<VarMap> var_map;
};

struct MouseHold: Node<AST::MouseHold> {
	using Node<AST::MouseHold>::Node;
	MouseButton button() const;
};

struct MouseRelease: Node<AST::MouseRelease> {
	using Node<AST::MouseRelease>::Node;
};

struct MouseWheel: Node<AST::MouseWheel> {
	using Node<AST::MouseWheel>::Node;
	std::string direction() const;
};

struct Plug: Node<AST::Plug> {
	using Node<AST::Plug>::Node;
	bool is_on() const;
};

struct PlugFlash: Node<AST::PlugFlash> {
	using Node<AST::PlugFlash>::Node;
	std::string name() const;
};

struct PlugNIC: Node<AST::PlugNIC> {
	using Node<AST::PlugNIC>::Node;
	std::string name() const;
};

struct PlugLink: Node<AST::PlugLink> {
	using Node<AST::PlugLink>::Node;
	std::string name() const;
};

struct PlugHostDev: Node<AST::PlugHostDev> {
	using Node<AST::PlugHostDev>::Node;
	std::string type() const;
	std::string addr() const;
};

struct PlugDVD: Node<AST::PlugDVD> {
	using Node<AST::PlugDVD>::Node;
	fs::path path() const;
};

struct Start:Node<AST::Start> {
	using Node<AST::Start>::Node;
};

struct Stop:Node<AST::Stop> {
	using Node<AST::Stop>::Node;
};

struct Shutdown:Node<AST::Shutdown> {
	using Node<AST::Shutdown>::Node;

	TimeInterval timeout() const;
};

struct Exec: Node<AST::Exec> {
	Exec(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
		Node(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_)) {}

	std::string interpreter() const;
	TimeInterval timeout() const;
	std::string script() const;

private:
	std::shared_ptr<VarMap> var_map;
};

struct Copy: Node<AST::Copy> {
	using Node<AST::Copy>::Node;
	TimeInterval timeout() const;
	std::string from() const;
	std::string to() const;
	bool nocheck() const;
};

struct Screenshot: Node<AST::Screenshot> {
	using Node<AST::Screenshot>::Node;
	std::string destination() const;
};

struct CycleControl: Node<AST::CycleControl> {
	using Node<AST::CycleControl>::Node;
	std::string type() const;
};

}
