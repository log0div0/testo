
#include "Base.hpp"
#include "../Exceptions.hpp"
#include "../Lexer.hpp"

namespace IR {

std::string SelectExpr::to_string() const {
	if (auto p = std::dynamic_pointer_cast<AST::SelectNegationExpr>(ast_node)) {
		return "!" + SelectExpr(p->expr, stack, var_map).to_string();
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectText>(ast_node)) {
		return String(p->str, stack, var_map).quoted_text();
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectJS>(ast_node)) {
		return p->token.value() + String(p->str, stack, var_map).quoted_text();
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectImg>(ast_node)) {
		return p->token.value() + String(p->str, stack, var_map).quoted_text();
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectParentedExpr>(ast_node)) {
		return "(" + SelectExpr(p->select_expr, stack, var_map).to_string() + ")";
	} else if (auto p = std::dynamic_pointer_cast<AST::SelectBinOp>(ast_node)) {
		return SelectExpr(p->left, stack, var_map).to_string() + " " + p->op.value() + " " + SelectExpr(p->right, stack, var_map).to_string();
	} else {
		throw std::runtime_error("Unknown select expression type");
	}
}

String::String(std::shared_ptr<ASTType> ast_node_, std::shared_ptr<StackNode> stack_):
	Node<AST::String>(std::move(ast_node_), std::move(stack_))
{
	if (ast_node->resolver.has_variables()) {
		throw ExceptionWithPos(ast_node->begin(), "Error: variables are not allowed in this context");
	}
}

String::String(std::shared_ptr<ASTType> ast_node, std::shared_ptr<StackNode> stack, std::shared_ptr<VarMap> var_map_):
	Node<AST::String>(std::move(ast_node), std::move(stack)), var_map(std::move(var_map_))
{
}

std::string String::text() const {
	try {
		return ast_node->resolver.resolve(stack, var_map);
	} catch (const std::exception& error) {
		std::throw_with_nested(ExceptionWithPos(ast_node->begin(), "Error while resolving \"" + ast_node->text() + "\""));
	}
}

std::string String::quoted_text() const {
	std::string str = text();
	{
		auto it = std::find(str.begin(), str.end(), '\n');
		if (it != str.end()) {
			return "\"\"\"" + str + "\"\"\"";
		}
	}
	{
		auto it = std::find(str.begin(), str.end(), '"');
		if (it != str.end()) {
			return "\"\"\"" + str + "\"\"\"";
		}
	}
	return "\"" + str + "\"";
}

nlohmann::json String::to_json() const {
	return text();
}

std::string String::str() const {
	return text();
}

bool String::can_resolve_variables() const {
	if (!ast_node->resolver.has_variables()) {
		return true;
	}
	return var_map != nullptr;
}

int32_t Number::value() const {
	return std::stoi(get_parsed()->to_string());
}

nlohmann::json Number::to_json() const {
	return value();
}

std::chrono::milliseconds time_to_milliseconds(const std::string& time) {
	uint32_t milliseconds;
	if (time[time.length() - 2] == 'm') {
		milliseconds = std::stoul(time.substr(0, time.length() - 2));
	} else if (time[time.length() - 1] == 's') {
		milliseconds = std::stoul(time.substr(0, time.length() - 1));
		milliseconds = milliseconds * 1000;
	} else if (time[time.length() - 1] == 'm') {
		milliseconds = std::stoul(time.substr(0, time.length() - 1));
		milliseconds = milliseconds * 1000 * 60;
	} else if (time[time.length() - 1] == 'h') {
		milliseconds = std::stoul(time.substr(0, time.length() - 1));
		milliseconds = milliseconds * 1000 * 60 * 60;
	} else {
		throw std::runtime_error("Unknown time specifier"); //should not happen ever
	}

	return std::chrono::milliseconds(milliseconds);
}

std::chrono::milliseconds TimeInterval::value() const {
	return time_to_milliseconds(get_parsed()->to_string());
}

size_t size_to_mb(const std::string& size) {
	size_t result = std::stoul(size.substr(0, size.length() - 2));
	if (size[size.length() - 2] == 'M') {
		result = result * 1;
	} else if (size[size.length() - 2] == 'G') {
		result = result * 1024;
	} else {
		throw Exception("Unknown size specifier"); //should not happen ever
	}

	return result;
}

size_t Size::megabytes() const {
	return size_to_mb(get_parsed()->to_string());
}

nlohmann::json Size::to_json() const {
	return megabytes();
}

bool str_to_bool(const std::string& str) {
	if (str == "true") {
		return true;
	} else if (str == "false") {
		return false;
	} else {
		throw std::runtime_error("Can't convert \"" + str + "\" to boolean");
	}
}

bool Boolean::value() const {
	return str_to_bool(get_parsed()->to_string());
}

nlohmann::json Boolean::to_json() const {
	return value();
}

std::string Id::value() const {
	return get_parsed()->to_string();
}

nlohmann::json Id::to_json() const {
	return value();
}

nlohmann::json AttrBlock::to_json() const {
	nlohmann::json config = nlohmann::json::object();
	for (auto attr: ast_node->attrs) {
		nlohmann::json j;
		if (auto p = std::dynamic_pointer_cast<AST::Number>(attr->value)) {
			j = Number(p, stack).to_json();
		} else if (auto p = std::dynamic_pointer_cast<AST::Size>(attr->value)) {
			j = Size(p, stack).to_json();
		} else if (auto p = std::dynamic_pointer_cast<AST::Boolean>(attr->value)) {
			j = Boolean(p, stack).to_json();
		} else if (auto p = std::dynamic_pointer_cast<AST::Id>(attr->value)) {
			j = Id(p, stack).to_json();
		} else if (auto p = std::dynamic_pointer_cast<AST::String>(attr->value)) {
			j = String(p, stack).to_json();
		} else if (auto p = std::dynamic_pointer_cast<AST::AttrBlock>(attr->value)) {
			j = AttrBlock(p, stack).to_json();
		} else {
			throw ExceptionWithPos(attr->begin(), "Error: Unsupported type of attr \"" + attr->name() + "\"");
		}
		if (attr->id) {
			j["name"] = attr->id.value();
			config[attr->name()].push_back(j);
		}  else {
			config[attr->name()] = j;
		}
	}
	return config;
}

std::string OptionSeq::resolve_param(const std::string& name) const {
	std::string param = stack->find_param(name);
	return template_literals::Resolver(param).resolve(stack, nullptr);
}

}
