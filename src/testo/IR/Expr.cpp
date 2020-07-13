
#include "Expr.hpp"
#include "Program.hpp"
#include "../TemplateLiterals.hpp"


namespace IR {

bool Defined::is_defined() const {
	return program->stack->is_defined(ast_node->var.value());
}

std::string Defined::var() const {
	return ast_node->var.value();
}


std::string Comparison::op() const {
	return ast_node->t.value();
}

std::string Comparison::left() const {
	return template_literals::Parser().resolve(ast_node->left->text(), stack);
}

std::string Comparison::right() const {
	return template_literals::Parser().resolve(ast_node->right->text(), stack);
}

std::vector<std::string> Range::values() const {
	std::vector<std::string> result;

	auto r1_num = std::stoul(r1());
	auto r2_num = std::stoul(r2());

	for (uint32_t i = r1_num; i < r2_num; ++i) {
		result.push_back(std::to_string(i));
	}

	return result;
}

std::string Range::r1() const {
	if (ast_node->r2) {
		return template_literals::Parser().resolve(ast_node->r1->text(), stack);
	} else {
		return "0";
	}
}

std::string Range::r2() const {
	if (ast_node->r2) {
		return template_literals::Parser().resolve(ast_node->r2->text(), stack);
	} else {
		return template_literals::Parser().resolve(ast_node->r1->text(), stack);
	}
}

}
