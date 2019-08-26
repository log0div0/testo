
#pragma once

#include <unordered_map>
#include <string>

struct StackEntry {
	StackEntry(bool is_terminate): is_terminate(is_terminate) {}

	void define(const std::string& name, const std::string& value) {
		vars[name] = value;
	}

	bool is_defined(const std::string& name) {
		return (vars.find(name) != vars.end());
	}

	std::string ref(const std::string& name) {
		auto found = vars.find(name);

		if (found != vars.end()) {
			return found->second;
		} else {
			throw std::runtime_error(std::string("Var ") + name + " not defined");
		}
	}

	bool is_terminate;
	std::unordered_map<std::string, std::string> vars;
};