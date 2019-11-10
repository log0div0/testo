
#include "VisitorSemantic.hpp"
#include <set>

namespace screen_selection {

VisitorSemantic::VisitorSemantic() {
	supported_columns.insert("TEXT");
	supported_columns.insert("FOREGROUND");
	supported_columns.insert("BACKROUND");


	supported_tables.insert("TEXTLINES");
}

void VisitorSemantic::visit(std::shared_ptr<AST::SelectStmt> select_stmt) {	

	if (!select_stmt->is_wildcard()) {
		for (auto column: select_stmt->columns) {
			if (supported_columns.find(column.value()) == supported_columns.end()) {
				throw std::runtime_error(std::string(column.pos()) + ": Error: unknown column name: " + column.value());
			}
		}
	}

	if (supported_tables.find(select_stmt->from_table.value()) == supported_tables.end()) {
		throw std::runtime_error(std::string(select_stmt->from_table.pos()) + ": Error: unknown table name: " + select_stmt->from_table.value());
	}

	visit_expr(select_stmt->where_expr);
}

void VisitorSemantic::visit_unop(std::shared_ptr<AST::UnOp> unop) {
	return visit_expr(unop->expr);
}

void VisitorSemantic::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	visit_expr(binop->left);
	visit_expr(binop->right);
}

void VisitorSemantic::visit_parented_expr(std::shared_ptr<AST::ParentedExpr> parented_expr) {
	return visit_expr(parented_expr->expr);
}

void VisitorSemantic::visit_factor(std::shared_ptr<AST::Factor> factor) {
	auto left = factor->left->t;
	auto right = factor->right->t;

	if (left.type() == Token::category::id) {
		if (supported_columns.find(left.value()) == supported_columns.end()) {
			throw std::runtime_error(std::string(left.pos()) + ": Error: unknown column name: " + left.value());
		}
	}

	if (right.type() == Token::category::id) {
		if (supported_columns.find(right.value()) == supported_columns.end()) {
			throw std::runtime_error(std::string(right.pos()) + ": Error: unknown column name: " + right.value());
		}
	}
}

void VisitorSemantic::visit_expr(std::shared_ptr<AST::IExpr> expr) {
	if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::BinOp>>(expr)) {
		return visit_binop(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::UnOp>>(expr)) {
		return visit_unop(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::ParentedExpr>>(expr)) {
		return visit_parented_expr(p->expr);
	} else if (auto p = std::dynamic_pointer_cast<AST::Expr<AST::Factor>>(expr)) {
		return visit_factor(p->expr);
	} else {
		throw std::runtime_error("Unknown expr type");
	}
}

}