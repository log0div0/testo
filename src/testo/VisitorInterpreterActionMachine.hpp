
#pragma once

#include <coro/Timer.h>
#include "VisitorInterpreterAction.hpp"

struct VisitorInterpreterActionMachine: public VisitorInterpreterAction {

	struct Point {
		int32_t x, y;
	};

	VisitorInterpreterActionMachine(std::shared_ptr<IR::Machine> vmc, std::shared_ptr<StackNode> stack, Reporter& reporter, std::shared_ptr<IR::Test> current_test);

	~VisitorInterpreterActionMachine() {}

	void visit_action(std::shared_ptr<AST::Action> action) override;
	void visit_copy(const IR::Copy& copy) override;
	bool visit_check(const IR::Check& check) override;
	void visit_abort(const IR::Abort& abort) override;

	void visit_type(const IR::Type& type);
	void visit_wait(const IR::Wait& wait);
	std::string visit_mouse_specifier_from(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier);
	std::string visit_mouse_specifier_centering(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier);
	std::string visit_mouse_specifier_default_centering();
	std::string visit_mouse_specifier_moving(std::shared_ptr<AST::MouseAdditionalSpecifier> specifier);
	std::string visit_mouse_additional_specifiers(const std::vector<std::shared_ptr<AST::MouseAdditionalSpecifier>>& specifiers);

	std::string build_select_text_script(const IR::SelectText& text);
	std::string build_select_img_script(const IR::SelectImg& img);

	bool visit_detect_js(const IR::SelectJS& js, const stb::Image<stb::RGB>& screenshot);
	bool visit_detect_expr(std::shared_ptr<AST::SelectExpr> select_expr, const stb::Image<stb::RGB>& screenshot);
	bool visit_detect_binop(std::shared_ptr<AST::SelectBinOp> binop, const stb::Image<stb::RGB>& screenshot);
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
	void visit_key_spec(const IR::KeySpec& key_spec, std::chrono::milliseconds interval);
	void visit_screenshot(const IR::Screenshot& screenshot);
	void visit_plug(const IR::Plug& plug);
	void visit_plug_nic(const IR::PlugNIC& plug_nic, bool is_on);
	void visit_plug_link(const IR::PlugLink& plug_link, bool is_on);
	void visit_plug_dvd(const IR::PlugDVD& plug_dvd);
	void visit_unplug_dvd(const IR::PlugDVD& plug_dvd);
	void visit_plug_flash(const IR::PlugFlash& plug_flash);
	void visit_unplug_flash(const IR::PlugFlash& plug_flash);
	void visit_plug_hostdev(const IR::PlugHostDev& plug_hostdev);
	void visit_unplug_hostdev(const IR::PlugHostDev& plug_hostdev);
	void visit_start(const IR::Start& start);
	void visit_stop(const IR::Stop& stop);
	void visit_shutdown(const IR::Shutdown& shutdown);
	void visit_exec(const IR::Exec& exec);

	nlohmann::json eval_js(const std::string& script, const stb::Image<stb::RGB>& screenshot);

	std::shared_ptr<IR::Machine> vmc;
	std::shared_ptr<IR::Test> current_test;
	coro::Timer timer;

private:
	template <typename Func>
	bool screenshot_loop(Func&& func, std::chrono::milliseconds timeout, std::chrono::milliseconds interval);
};