
#pragma once

#include "Node.hpp"

namespace screen_selection{

struct VisitorInterpreter {
	VisitorInterpreter();

	void visit(std::shared_ptr<AST::SelectStmt> select_stmt);
	void visit_expr(std::shared_ptr<AST::IExpr> expr);
	void visit_unop(std::shared_ptr<AST::UnOp> unop);
	void visit_binop(std::shared_ptr<AST::BinOp> binop);
	void visit_parented_expr(std::shared_ptr<AST::ParentedExpr> parented_expr);
	void visit_factor(std::shared_ptr<AST::Factor> factor);

private:
	std::string text, background, foreground;
};

}
