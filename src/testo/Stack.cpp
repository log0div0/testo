
#include "Stack.hpp"
#include "TemplateLiterals.hpp"

std::string StackNode::find_and_resolve_var(const std::string& name) const {
	auto it = vars.find(name);
	if (it != vars.end()) {
		return template_literals::Parser().resolve(it->second, shared_from_this());
	}
	if (!parent) {
		throw std::runtime_error("param \"" + name + "\" is not defined");
	}
	return parent->find_and_resolve_var(name);
}

bool StackNode::is_defined(const std::string& var) const {
	auto it = vars.find(var);
	if (it != vars.end()) {
		return true;
	}
	if (!parent) {
		return false;
	}
	return parent->is_defined(var);
}