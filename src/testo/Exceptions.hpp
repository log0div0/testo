
#include "AST.hpp"
#include "IR/Machine.hpp"
#include <stdexcept>
#include <string>

struct InterpreterException: public std::exception {
	explicit InterpreterException():
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

struct ActionException: public InterpreterException {
	explicit ActionException(std::shared_ptr<AST::Node> node, std::shared_ptr<IR::Machine> vmc):
		InterpreterException()
	{
		msg = std::string(node->begin()) + ": Error while performing action " + std::string(*node);
		if (vmc) {
			msg += " on virtual machine ";
			msg += vmc->name();
		}
	}
};

struct MacroException: public InterpreterException {
	explicit MacroException(std::shared_ptr<AST::MacroCall> macro_call):
		InterpreterException()
	{
		msg = std::string(macro_call->begin()) + std::string(": In a macro call ") + macro_call->name().value();
	}
};

struct AbortException: public InterpreterException {
	explicit AbortException(std::shared_ptr<AST::Abort> node, std::shared_ptr<IR::Machine> vmc, const std::string& message):
		InterpreterException()
	{
		msg = std::string(node->begin()) + ": Caught abort action ";
		if (vmc) {
			msg += "on virtual machine ";
			msg += vmc->name();
		}

		msg += " with message: ";
		msg += message;
	}
};


struct CycleControlException: public InterpreterException {
	explicit CycleControlException(const Token& token):
		InterpreterException(), token(token)
	{
		msg = std::string(token.begin()) + " error: cycle control action has not a correcponding cycle";
	}

	Token token;
};

static void backtrace(std::ostream& stream, const std::exception& error) {
	stream << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const MacroException& error) {
		stream << "\n";
		backtrace(stream, error);
	} catch (const InterpreterException& error) {
		stream << "\n";
		backtrace(stream, error);
	} catch (const std::exception& error) {
		stream << "\n\t-";
		backtrace(stream, error);
	} catch(...) {
		stream << std::endl;
		stream << "[Unknown exception type]";
	}
}

inline std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error);
	return stream;
}