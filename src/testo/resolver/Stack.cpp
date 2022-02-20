
#include "Stack.hpp"

std::string StackNode::find_param(const std::string& name) const {
	auto it = params.find(name);
	if (it != params.end()) {
		return it->second;
	}
	if (!parent) {
		throw std::runtime_error("param \"" + name + "\" is not defined");
	}
	return parent->find_param(name);
}

bool StackNode::is_defined(const std::string& var) const {
	auto it = params.find(var);
	if (it != params.end()) {
		return true;
	}
	if (!parent) {
		return false;
	}
	return parent->is_defined(var);
}
