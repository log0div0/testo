
#pragma once

#include "Base.hpp"
#include "../Mouse.hpp"
#include "../Keyboard.hpp"

namespace IR {

struct KeyCombination: MaybeUnparsed<AST::IKeyCombination, AST::KeyCombination> {
	using MaybeUnparsed::MaybeUnparsed;
	std::vector<KeyboardButton> buttons() const;
	std::string to_string() const;
};

template <typename ASTType>
struct Action: Node<ASTType> {
	using Node<ASTType>::Node;
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
	TimeInterval interval() const;
};

struct KeySpec: Action<AST::KeySpec> {
	using Action<AST::KeySpec>::Action;
	KeyCombination combination() const;
	int32_t times() const;
};

struct Hold: Action<AST::Hold> {
	using Action<AST::Hold>::Action;
	KeyCombination combination() const;
};

struct Release: Action<AST::Release> {
	using Action<AST::Release>::Action;
	KeyCombination combination() const;
};

struct Type: Action<AST::Type> {
	using Action<AST::Type>::Action;
	std::string text() const;
	TimeInterval interval() const;
	KeyCombination autoswitch() const;
	bool use_autoswitch() const;
};

struct Wait: Action<AST::Wait> {
	using Action<AST::Wait>::Action;
	SelectExpr select_expr() const;
	TimeInterval timeout() const;
	TimeInterval interval() const;
};

struct Sleep: Action<AST::Sleep> {
	using Action<AST::Sleep>::Action;
	TimeInterval timeout() const;
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
	std::string to_string() const;
	TimeInterval timeout() const;
};

struct SelectJS: Action<AST::SelectJS> {
	using Action<AST::SelectJS>::Action;
	std::string script() const;
};

struct SelectImg: Action<AST::SelectImg> {
	using Action<AST::SelectImg>::Action;
	fs::path img_path() const;
};

struct SelectText: Action<AST::SelectText> {
	using Action<AST::SelectText>::Action;
	std::string text() const;
};

struct MouseHold: Action<AST::MouseHold> {
	using Action<AST::MouseHold>::Action;
	MouseButton button() const;
};

struct MouseRelease: Action<AST::MouseRelease> {
	using Action<AST::MouseRelease>::Action;
};

struct MouseWheel: Action<AST::MouseWheel> {
	using Action<AST::MouseWheel>::Action;
	std::string direction() const;
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

	TimeInterval timeout() const;
};

struct Exec: Action<AST::Exec> {
	using Action<AST::Exec>::Action;
	std::string interpreter() const;
	TimeInterval timeout() const;
	std::string script() const;
};

struct Copy: Action<AST::Copy> {
	using Action<AST::Copy>::Action;
	TimeInterval timeout() const;
	std::string from() const;
	std::string to() const;
	bool nocheck() const;
};

struct Screenshot: Action<AST::Screenshot> {
	using Action<AST::Screenshot>::Action;
	std::string destination() const;
};

struct CycleControl: Action<AST::CycleControl> {
	using Action<AST::CycleControl>::Action;
	std::string type() const;
};

}
