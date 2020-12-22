
#pragma once

#include <coro/Timer.h>
#include "VisitorInterpreterAction.hpp"
#include <nn/TextTensor.hpp>
#include <nn/ImgTensor.hpp>
#include <nn/Homm3Tensor.hpp>
#include "js/Context.hpp"

struct VisitorInterpreterActionMachine: public VisitorInterpreterAction {
	VisitorInterpreterActionMachine(std::shared_ptr<IR::Machine> vmc, std::shared_ptr<StackNode> stack, Reporter& reporter, std::shared_ptr<IR::Test> current_test);

	~VisitorInterpreterActionMachine() {}

	void visit_action(std::shared_ptr<AST::IAction> action) override;
	void visit_copy(const IR::Copy& copy) override;
	bool visit_check(const IR::Check& check) override;
	void visit_abort(const IR::Abort& abort) override;

	void visit_type(const IR::Type& type);
	void visit_wait(const IR::Wait& wait);
	template <typename NNTensor>
	NNTensor visit_mouse_specifier_from(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const NNTensor& input);
	template <typename NNTensor>
	nn::Point visit_mouse_specifier_centering(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const NNTensor& input);
	template <typename NNTensor>
	nn::Point visit_mouse_specifier_default_centering(const NNTensor& input);
	nn::Point visit_mouse_specifier_moving(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier, const nn::Point& input);
	template <typename NNTensor>
	nn::Point visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers, const NNTensor& input);
	nn::TextTensor visit_select_text(const IR::SelectText& text, stb::Image<stb::RGB>& screenshot);
	nn::ImgTensor visit_select_img(const IR::SelectImg& img, stb::Image<stb::RGB>& screenshot);
	nn::Homm3Tensor visit_select_homm3(const IR::SelectHomm3& homm3, stb::Image<stb::RGB>& screenshot);
	bool visit_detect_js(const IR::SelectJS& js, stb::Image<stb::RGB>& screenshot);
	nn::Point visit_select_js(const IR::SelectJS& js, stb::Image<stb::RGB>& screenshot);
	bool visit_detect_expr(std::shared_ptr<AST::ISelectExpr> select_expr, stb::Image<stb::RGB>& screenshot);
	bool visit_detect_selectable(std::shared_ptr<AST::ISelectable> selectable, stb::Image<stb::RGB>& screenshot);
	bool visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop, stb::Image<stb::RGB>& screenshot);
	void visit_press(const IR::Press& press);
	void visit_hold(const IR::Hold& hold);
	void visit_release(const IR::Release& release);
	void visit_mouse_move_selectable(const IR::MouseSelectable& mouse_selectable);
	void visit_mouse(const IR::Mouse& mouse);
	void visit_mouse_move_click(const IR::MouseMoveClick& mouse_move_click);
	void visit_mouse_move_coordinates(const IR::MouseCoordinates& coordinates);
	void visit_mouse_hold(const IR::MouseHold& mouse_hold);
	void visit_mouse_release(const IR::MouseRelease& mouse_release);
	void visit_mouse_wheel(std::shared_ptr<AST::MouseWheel> mouse_wheel);
	void visit_key_spec(std::shared_ptr<AST::KeySpec> key_spec, uint32_t interval);
	void visit_plug(const IR::Plug& plug);
	void visit_plug_nic(const IR::Plug& plug);
	void visit_plug_link(const IR::Plug& plug);
	void visit_plug_dvd(const IR::Plug& plug);
	void visit_plug_flash(const IR::Plug& plug);
	void visit_unplug_flash(const IR::Plug& plug);
	void visit_start(const IR::Start& start);
	void visit_stop(const IR::Stop& stop);
	void visit_shutdown(const IR::Shutdown& shutdown);
	void visit_exec(const IR::Exec& exec);

	js::Value eval_js(const std::string& script, stb::Image<stb::RGB>& screenshot);

	std::shared_ptr<IR::Machine> vmc;
	std::shared_ptr<IR::Test> current_test;
	coro::Timer timer;
	std::unordered_map<char32_t, std::vector<std::string>> charmap;
	std::shared_ptr<js::Context> js_current_ctx;
};