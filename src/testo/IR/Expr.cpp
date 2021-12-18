
#include "Program.hpp"
#include "Expr.hpp"
#include "Action.hpp"
#include "OptionSeq.hpp"
#include "../Utils.hpp"
#include "../TemplateLiterals.hpp"


namespace IR {

bool Defined::is_defined() const {
	return program->stack->is_defined(ast_node->var.value());
}

std::string Defined::var() const {
	return ast_node->var.value();
}

std::string Comparison::op() const {
	return ast_node->op.value();
}

std::string Comparison::left() const {
	return String(ast_node->left, stack).text();
}

std::string Comparison::right() const {
	return String(ast_node->right, stack).text();
}

bool Comparison::calculate() const {
	auto l = left();
	auto r = right();
	if (op() == "GREATER") {
		if (!is_number(l)) {
			throw ExceptionWithPos(ast_node->left->begin(), "Error: \"" + l + "\" is not an integer number");
		}
		if (!is_number(r)) {
			throw ExceptionWithPos(ast_node->right->begin(), "Error: \"" + r + "\" is not an integer number");
		}

		return std::stoul(l) > std::stoul(r);

	} else if (op() == "LESS") {
		if (!is_number(l)) {
			throw ExceptionWithPos(ast_node->left->begin(), "Error: \"" + l + "\" is not an integer number");
		}
		if (!is_number(r)) {
			throw ExceptionWithPos(ast_node->right->begin(), "Error: \"" + r + "\" is not an integer number");
		}

		return std::stoul(l) < std::stoul(r);

	} else if (op() == "EQUAL") {
		if (!is_number(l)) {
			throw ExceptionWithPos(ast_node->left->begin(), "Error: \"" + l + "\" is not an integer number");
		}
		if (!is_number(r)) {
			throw ExceptionWithPos(ast_node->right->begin(), "Error: \"" + r + "\" is not an integer number");
		}

		return std::stoul(l) == std::stoul(r);

	} else if (op() == "STRGREATER") {
		return l > r;
	} else if (op() == "STRLESS") {
		return l < r;
	} else if (op() == "STREQUAL") {
		return l == r;
	} else {
		throw std::runtime_error("Unknown comparison op");
	}
}

std::string StringExpr::text() const {
	return String(ast_node->str, stack).text();
}

SelectExpr Check::select_expr() const {
	return SelectExpr(ast_node->select_expr, stack, var_map);
}

TimeInterval Check::timeout() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("timeout", "TESTO_CHECK_DEFAULT_TIMEOUT");
}

TimeInterval Check::interval() const {
	return OptionSeq(ast_node->option_seq, stack).get<TimeInterval>("interval", "TESTO_CHECK_DEFAULT_INTERVAL");
}

std::vector<std::string> Range::values() const {
	std::vector<std::string> result;

	auto r1_num = r1();
	auto r2_num = r2();

	for (int32_t i = r1_num; i < r2_num; ++i) {
		result.push_back(std::to_string(i));
	}

	return result;
}

int32_t Range::r1() const {
	if (ast_node->r2) {
		return Number(ast_node->r1, stack).value();
	} else {
		return 0;
	}
}

int32_t Range::r2() const {
	if (ast_node->r2) {
		return Number(ast_node->r2, stack).value();
	} else {
		return Number(ast_node->r1, stack).value();
	}
}

}
