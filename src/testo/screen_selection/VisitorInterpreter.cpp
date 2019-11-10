
#include "VisitorInterpreter.hpp"
#include <set>

namespace screen_selection {

VisitorInterpreter::VisitorInterpreter() {}

void VisitorInterpreter::visit(std::shared_ptr<AST::SelectStmt> select_stmt) {	
	visit_expr(select_stmt->where_expr);
}

void VisitorInterpreter::visit_unop(std::shared_ptr<AST::UnOp> unop) {
	return visit_expr(unop->expr);
}

void VisitorInterpreter::visit_binop(std::shared_ptr<AST::BinOp> binop) {
	visit_expr(binop->left);
	if (binop->t.type() != Token::category::and_) {
		throw std::runtime_error(std::string(binop->t.pos()) + ": Error: unsupported operation: " + binop->t.value());
	}
	visit_expr(binop->right);
}

void VisitorInterpreter::visit_parented_expr(std::shared_ptr<AST::ParentedExpr> parented_expr) {
	return visit_expr(parented_expr->expr);
}

void VisitorInterpreter::visit_factor(std::shared_ptr<AST::Factor> factor) {
	auto left = factor->left->t;
	auto right = factor->right->t;

	if (factor->op.type() != Token::category::equals) {
		throw std::runtime_error(std::string(factor->op.pos()) + ": Error: unsupported operation: " + factor->op.value());
	}
}

void VisitorInterpreter::visit_expr(std::shared_ptr<AST::IExpr> expr) {
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