
#pragma once

#include <memory>
#include <map>

struct StackNode {
	StackNode() = default;

	StackNode(const StackNode& other) = delete;
	StackNode& operator=(const StackNode& other) = delete;
	StackNode(StackNode&& other) = delete;
	StackNode& operator=(StackNode&& other) = delete;

	std::string resolve_var(const std::string& name) const {
		auto it = vars.find(name);
		if (it != vars.end()) {
			return it->second;
		}
		if (!parent) {
			throw std::runtime_error(std::string("Var ") + name + " not defined");
		}
		return parent->resolve_var(name);
	}

	std::shared_ptr<StackNode> parent;
	std::map<std::string, std::string> vars;
};

template <typename StackHolder>
struct StackPusher {
	StackPusher(StackHolder* stack_holder_, std::shared_ptr<StackNode> new_stack): stack_holder(stack_holder_) {
		backup = std::move(stack_holder->stack);
		stack_holder->stack = std::move(new_stack);
	}
	~StackPusher() {
		stack_holder->stack = std::move(backup);
	}

	StackPusher(const StackPusher& other) = delete;
	StackPusher& operator=(const StackPusher& other) = delete;
	StackPusher(StackPusher&& other) = delete;
	StackPusher& operator=(StackPusher&& other) = delete;

	StackHolder* stack_holder = nullptr;
	std::shared_ptr<StackNode> backup;
};
