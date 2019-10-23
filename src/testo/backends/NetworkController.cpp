
#include "NetworkController.hpp"
#include "Environment.hpp"
#include <fmt/format.h>

std::string NetworkController::id() const {

}

std::string NetworkController::name() const {

}

bool NetworkController::is_defined() const {

}

void NetworkController::create() {

}

void NetworkController::create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed)
{

}

void NetworkController::restore_snapshot(const std::string& snapshot) {

}

void NetworkController::delete_snapshot_with_children(const std::string& snapshot)
{

}

bool NetworkController::has_user_key(const std::string& key) {

}


std::string NetworkController::get_user_metadata(const std::string& key) {

}

void NetworkController::set_user_metadata(const std::string& key, const std::string& value) {

}

bool NetworkController::check_config_relevance() {

}

fs::path NetworkController::get_metadata_dir() const {
}

