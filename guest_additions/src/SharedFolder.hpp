
#pragma once

#include <nlohmann/json.hpp>
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

fs::path get_config_path();
nlohmann::json load_config();
void save_config(const nlohmann::json& config);
void register_shared_folder(const std::string& folder_name, const fs::path& guest_path);
void unregister_shared_folder(const std::string& folder_name);
nlohmann::json get_shared_folder_status(const std::string& folder_name);
bool mount_shared_folder(const std::string& folder_name, const fs::path& guest_path);
bool umount_shared_folder(const std::string& folder_name);
void mount_permanent_shared_folders();
