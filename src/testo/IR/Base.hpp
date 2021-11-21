
#pragma once

#include "../AST.hpp"
#include "../Stack.hpp"
#include "../Parser.hpp"
#include "../Exceptions.hpp"

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

struct SelectExpr: Node<AST::SelectExpr> {
	using Node<AST::SelectExpr>::Node;

	std::string to_string() const;
};

struct String: Node<AST::String> {
	String(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, bool variables_allowed = false);

	std::string text() const;
	std::string quoted_text() const;
	nlohmann::json to_json() const;

	std::string str() const;
	bool can_resolve_variables() const;
};

template <typename ASTType, typename ParsedASTType>
struct MaybeUnparsed: Node<ASTType> {
	using Node<ASTType>::Node;

	std::shared_ptr<ParsedASTType> get_parsed() const {
		if (_parsed) {
			return _parsed;
		}
		_parsed = std::dynamic_pointer_cast<ParsedASTType>(this->ast_node);
		if (_parsed) {
			return _parsed;
		}
		auto unparsed = std::dynamic_pointer_cast<AST::Unparsed<ASTType>>(this->ast_node);
		if (!unparsed) {
			throw std::runtime_error("Failed to cast AST node to any derived class");
		}
		String str(unparsed->string, this->stack);
		std::shared_ptr<ASTType> p = parse(str);
		_parsed = std::dynamic_pointer_cast<ParsedASTType>(p);
		if (!_parsed) {
			throw std::runtime_error("Failed to cast AST node to the parsed derived class");
		}
		return _parsed;
	}

private:
	std::shared_ptr<ASTType> parse(const String& str) const {
		try {
			try {
				return ASTType::from_string(str.text());
			} catch (const ExceptionWithPos& error) {
				// trim information about the position, because it's totaly useless
				// when we parse a string literal
				throw std::runtime_error(error.original_msg);
			}
		} catch (const std::exception& error) {
			std::throw_with_nested(ExceptionWithPos(str.ast_node->begin(), "Error while parsing " + str.quoted_text()));
		}
	}

	mutable std::shared_ptr<ParsedASTType> _parsed;
};

template <Token::category category>
struct SingleToken: MaybeUnparsed<AST::ISingleToken<category>, AST::SingleToken<category>> {
	using MaybeUnparsed<AST::ISingleToken<category>, AST::SingleToken<category>>::MaybeUnparsed;

	std::string str() const {
		return this->get_parsed()->to_string();
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
				throw std::runtime_error("Failed to cast \"" + name + "\" option to the target type");
			}
		} else {
			std::string str = get_param(default_param);
			p = T::ASTType::from_string(str);
		}
		return {p, stack};
	}

	template <typename T>
	T get(const std::string& name) const {
		std::shared_ptr<typename T::ASTType> p = nullptr;
		if (auto option = ast_node->get(name)) {
			p = std::dynamic_pointer_cast<typename T::ASTType>(option->value);
			if (!p) {
				throw std::runtime_error("Failed to cast \"" + name + "\" option to the target type");
			}
			return {p, stack};
		} else {
			throw std::runtime_error("Option \"" + name + "\" is not specified");
		}
	}

	bool has(const std::string& name) {
		return ast_node->get(name) != nullptr;
	}

private:
	static std::string get_param(const std::string& name);
};

}
