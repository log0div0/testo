
#pragma once

#include "Stack.hpp"
#include "AST.hpp"

struct VisitorAttr {
	VisitorAttr(std::shared_ptr<StackNode> stack_): stack(std::move(stack_)) {}
	std::shared_ptr<StackNode> stack;

protected:
	template <typename Visitor>
	nlohmann::json visit_attr_block(std::shared_ptr<AST::Attr> attr, Visitor&& visitor);
	template <typename Visitor>
	nlohmann::json visit_attr_block(std::shared_ptr<AST::AttrBlock> attr_block, Visitor&& visitor);

	nlohmann::json visit_attr_simple_value(std::shared_ptr<AST::Attr> attr, Token::category category);
	nlohmann::json visit_attr_simple_value(std::shared_ptr<AST::AttrSimpleValue> attr_simple_value, Token::category category);
};

struct VisitorAttrMachine: VisitorAttr {
	using VisitorAttr::VisitorAttr;

	nlohmann::json visit(std::shared_ptr<AST::AttrBlock> attr_block);

private:
	nlohmann::json visit_nic(std::shared_ptr<AST::Attr> attr);
	nlohmann::json visit_disk(std::shared_ptr<AST::Attr> attr);
	nlohmann::json visit_video(std::shared_ptr<AST::Attr> attr);
	nlohmann::json visit_shared_folder(std::shared_ptr<AST::Attr> attr);
};

struct VisitorAttrFlashDrive: VisitorAttr {
	using VisitorAttr::VisitorAttr;

	nlohmann::json visit(std::shared_ptr<AST::AttrBlock> attr_block);
};

struct VisitorAttrNetwork: VisitorAttr {
	using VisitorAttr::VisitorAttr;

	nlohmann::json visit(std::shared_ptr<AST::AttrBlock> attr_block);
};

struct VisitorAttrTest: VisitorAttr {
	using VisitorAttr::VisitorAttr;

	nlohmann::json visit(std::shared_ptr<AST::AttrBlock> attr_block);
};
