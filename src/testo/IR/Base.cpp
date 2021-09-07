
#include "Base.hpp"
#include "Program.hpp"
#include "../TemplateLiterals.hpp"
#include "../Exceptions.hpp"
#include "../Lexer.hpp"

struct ResolveException: public Exception {
	ResolveException(const Pos& pos, const std::string& string)
	{
		msg = std::string(pos) + ": Error while resolving \"" + string + "\"";
	}
};

namespace IR {

std::string String::text() const {
	try {
		return template_literals::Parser().resolve(ast_node->text(), stack);
	} catch (const std::exception& error) {
		std::throw_with_nested(ResolveException(ast_node->begin(), ast_node->text()));
	}
}

nlohmann::json String::to_json() const {
	return text();
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
			throw std::runtime_error(std::string(attr->begin()) + ": Error: Unsupported type of attr \"" + attr->name() + "\"");
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

std::string OptionSeq::get_param(const std::string& name) {
	return program->stack->find_and_resolve_var(name);
}

}
