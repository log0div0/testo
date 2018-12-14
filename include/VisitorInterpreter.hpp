
#pragma once

#include <Node.hpp>
#include <Global.hpp>
#include <VmController.hpp>

struct VisitorInterpreter {
	VisitorInterpreter(Global& global):
		global(global) {}

	void visit(std::shared_ptr<AST::Program> program);
	void visit_stmt(std::shared_ptr<AST::IStmt> stmt);
	void visit_controller(std::shared_ptr<AST::Controller> controller);
	void visit_flash(std::shared_ptr<AST::Controller> flash);
	void visit_snapshot(std::shared_ptr<AST::Snapshot> snapshot);
	void visit_test(std::shared_ptr<AST::Test> test);
	void visit_vm_state(std::shared_ptr<AST::VmState> vm_state);
	void visit_command_block(std::shared_ptr<AST::CmdBlock> block);
	void visit_command(std::shared_ptr<AST::Cmd> cmd);
	void visit_action_block(std::shared_ptr<VmController> vm, std::shared_ptr<AST::ActionBlock> action_block);
	void visit_action(std::shared_ptr<VmController> vm, std::shared_ptr<AST::IAction> action);
	void visit_type(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Type> type);
	void visit_wait(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Wait> wait);
	void visit_press(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Press> press);
	void visit_key_spec(std::shared_ptr<VmController> vm, std::shared_ptr<AST::KeySpec> key_spec);
	void visit_plug(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void visit_plug_nic(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void visit_plug_dvd(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void plug_flash(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void unplug_flash(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Plug> plug);
	void visit_start(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Start> start);
	void visit_stop(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Stop> stop);
	void visit_exec(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Exec> exec);
	void visit_set(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Set> set);
	void visit_copyto(std::shared_ptr<VmController> vm, std::shared_ptr<AST::CopyTo> copyto);
	
	void apply_actions(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Snapshot> snapshot, bool recursive = false);
	bool resolve_state(std::shared_ptr<VmController> vm, std::shared_ptr<AST::Snapshot> snapshot);
	std::string cksum(std::shared_ptr<AST::Snapshot> snapshot);
	std::string cksum(std::shared_ptr<AST::Controller> controller);

	Global& global;
};
