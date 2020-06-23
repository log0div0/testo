
#pragma once

#include "Register.hpp"

std::set<std::shared_ptr<Controller>> get_all_controllers(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg);
std::set<std::shared_ptr<VmController>> get_all_vmcs(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg);
std::set<std::shared_ptr<NetworkController>> get_all_netcs(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg);
std::set<std::shared_ptr<FlashDriveController>> get_all_fdcs(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg);
std::list<std::shared_ptr<AST::Test>> get_test_path(std::shared_ptr<AST::Test> test, std::shared_ptr<Register> reg);
std::set<std::shared_ptr<FlashDriveController>> extract_fdcs_from_action(std::shared_ptr<AST::IAction> action, std::shared_ptr<Register> reg);