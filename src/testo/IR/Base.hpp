
#pragma once

#include "../AST.hpp"
#include "../Stack.hpp"
#include "../TemplateLiterals.hpp"
#include "../Parser.hpp"

namespace IR {

template <typename ASTType_>
struct Node {
	using ASTType = ASTType_;

	Node() = default;
	Node(std::shared_ptr<ASTType> ast_node_, std::shared_ptr<StackNode> stack_):
		ast_node(std::move(ast_node_)), stack(std::move(stack_)) {}
	std::shared_ptr<ASTType> ast_node;
	std::shared_ptr<StackNode> stack;
};

struct String: Node<AST::String> {
	using Node::Node;

	std::string text() const;
	nlohmann::json to_json() const;
};

template <typename IASTType, typename ASTType>
struct MaybeUnparsed: Node<IASTType> {
	using Node::Node;

protected:
	std::shared_ptr<ASTType> get_parsed() const {
		if (_parsed) {
			return _parsed;
		}
		_parsed = std::dynamic_pointer_cast<ASTType>(ast_node);
		if (_parsed) {
			return _parsed;
		}
		auto unparsed = std::dynamic_pointer_cast<AST::Unparsed<IASTType>>(ast_node);
		if (!unparsed) {
			throw std::runtime_error("Failed to cast AST node to any derived class");
		}
		std::string str = unparsed->string->text();
		str = template_literals::Parser().resolve(str, stack);
		std::shared_ptr<IASTType> p = IASTType::from_string(str);
		_parsed = std::dynamic_pointer_cast<ASTType>(p);
		if (!_parsed) {
			throw std::runtime_error("Failed to cast AST node to the parsed derived class");
		}
		return _parsed;
	}

private:
	mutable std::shared_ptr<ASTType> _parsed;
};

template <Token::category category>
struct SingleToken: MaybeUnparsed<AST::ISingleToken<category>, AST::SingleToken<category>> {
	using MaybeUnparsed::MaybeUnparsed;

	std::string str() const {
		return get_parsed()->to_string();
	}
};

struct Number: SingleToken<Token::category::number> {
	using SingleToken::SingleToken;

	int32_t value() const;
	nlohmann::json to_json() const;
};

struct TimeInterval: SingleToken<Token::category::time_interval> {
	using SingleToken::SingleToken;

	std::chrono::milliseconds value() const;
};

struct Size: SingleToken<Token::category::size> {
	using SingleToken::SingleToken;

	size_t megabytes() const;
	nlohmann::json to_json() const;
};

struct Boolean: SingleToken<Token::category::boolean> {
	using SingleToken::SingleToken;

	bool value() const;
	nlohmann::json to_json() const;
};

struct Id: SingleToken<Token::category::id> {
	using SingleToken::SingleToken;

	std::string value() const;
	nlohmann::json to_json() const;
};

struct AttrBlock: Node<AST::AttrBlock> {
	using Node::Node;

	nlohmann::json to_json() const;
};

struct OptionSeq : Node<AST::OptionSeq> {
	using Node::Node;

	template <typename T>
	T get(const std::string& name, const std::string& default_param) const {
		std::shared_ptr<typename T::ASTType> p = nullptr;
		if (auto option = ast_node->get(name)) {
			p = std::dynamic_pointer_cast<typename T::ASTType>(option->value);
			if (!p) {
				throw std::runtime_error("Failed to cast \'" + name + "\" option to the target type");
			}
		} else {
			std::string str = get_param(default_param);
			p = T::ASTType::from_string(str);
		}
		return {p, stack};
	}

	bool has(const std::string& name) {
		return ast_node->get(name) != nullptr;
	}

private:
	static std::string get_param(const std::string& name);
};

}
