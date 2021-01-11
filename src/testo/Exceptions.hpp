
#include "AST.hpp"
#include "IR/Machine.hpp"
#include <stdexcept>
#include <string>

struct Exception: public std::exception {
	explicit Exception():
		std::exception()
	{
		msg = "";
	}

	const char* what() const noexcept override {
		return msg.c_str();
	}
protected:
	std::string msg;
};

struct TestFailedException: public Exception {
	explicit TestFailedException():
		Exception()
	{
		msg = "At least one of the tests failed";
	}
};

struct ActionException: public Exception {
	explicit ActionException(std::shared_ptr<AST::Node> node, std::shared_ptr<IR::Controller> controller):
		Exception()
	{
		msg = std::string(node->begin()) + ": Error while performing action " + std::string(*node);
		if (controller) {
			msg += " on " + controller->type() + " " + controller->name();
		}
	}
};

struct MacroException: public Exception {
	explicit MacroException(std::shared_ptr<AST::MacroCall> macro_call):
		Exception()
	{
		msg = std::string(macro_call->begin()) + std::string(": In a macro call ") + macro_call->name().value();
	}
};

struct AbortException: public Exception {
	explicit AbortException(std::shared_ptr<AST::Abort> node, std::shared_ptr<IR::Controller> controller, const std::string& message):
		Exception()
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
	explicit CycleControlException(const Token& token):
		Exception(), token(token)
	{
		msg = std::string(token.begin()) + " error: cycle control action has not a correcponding cycle";
	}

	Token token;
};

struct ResolveException: public Exception {
	explicit ResolveException(const Pos& pos, const std::string& string):
		Exception()
	{
		msg = std::string(pos) + ": Error while resolving \"" + string + "\"";
	}
};

std::ostream& operator<<(std::ostream& stream, const std::exception& error);
