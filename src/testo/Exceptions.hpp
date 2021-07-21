
#pragma once

#include "AST.hpp"
#include "IR/Machine.hpp"
#include <stdexcept>
#include <string>

struct Exception: public std::exception {
	Exception() = default;
	Exception(const std::string& what) {
		msg = what;
	}
	const char* what() const noexcept override {
		return msg.c_str();
	}
protected:
	std::string msg;
};

struct TestFailedException: public Exception {
	TestFailedException()
	{
		msg = "At least one of the tests failed";
	}
};

struct ControllerCreatonException: public Exception {
	ControllerCreatonException(std::shared_ptr<IR::Controller> controller) {
		std::stringstream ss;
		for (auto macro_call: controller->macro_call_stack) {
			ss << std::string(macro_call->begin()) + std::string(": In a macro call ") << macro_call->name().value() << std::endl;
		}

		ss << std::string(controller->ast_node->begin()) << ": In the " << controller->type() << " \"" << controller->name() << "\" declaration";
		msg = ss.str();
	}
};

struct ActionException: public Exception {
	ActionException(std::shared_ptr<AST::Node> node, std::shared_ptr<IR::Controller> controller): controller(controller)
	{
		msg = std::string(node->begin()) + ": Error while performing action " + std::string(*node);
		if (controller) {
			msg += " on " + controller->type() + " " + controller->name();
		}
	}

	std::shared_ptr<IR::Controller> controller;
};

struct MacroException: public Exception {
	MacroException(std::shared_ptr<AST::MacroCall> macro_call)
	{
		msg = std::string(macro_call->begin()) + std::string(": In a macro call ") + macro_call->name().value();
	}
};

struct AbortException: public Exception {
	AbortException(std::shared_ptr<AST::Abort> node, std::shared_ptr<IR::Controller> controller, const std::string& message)
	{
		msg = std::string(node->begin()) + ": Caught abort action ";
		if (controller) {
			msg += "on " + controller->type() + " " +  controller->name();
		}

		msg += " with message: ";
		msg += message;
	}
};


struct CycleControlException: public Exception {
	CycleControlException(const Token& token): token(token)
	{
		msg = std::string(token.begin()) + " error: cycle control action has not a correcponding cycle";
	}

	Token token;
};

struct ResolveException: public Exception {
	ResolveException(const Pos& pos, const std::string& string)
	{
		msg = std::string(pos) + ": Error while resolving \"" + string + "\"";
	}
};

struct ContinueError: std::runtime_error {
	using std::runtime_error::runtime_error;
};

std::ostream& operator<<(std::ostream& stream, const std::exception& error);
