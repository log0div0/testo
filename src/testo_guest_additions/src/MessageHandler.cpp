
#include "MessageHandler.hpp"
#include <os/Process.hpp>
#include <os/File.hpp>

#include <base64.hpp>
#include <scope_guard.hpp>
#include <spdlog/spdlog.h>

#include <stdexcept>
#include <fstream>

#ifdef WIN32
#include <winapi/Functions.hpp>
#include <winapi/RegKey.hpp>

std::map<std::string, std::string> get_environment_from_registry() {
	std::map<std::string, std::string> result;
	winapi::RegKey regkey(HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment");
	for (const std::string& name: regkey.enum_values()) {
		result[name] = regkey.get_str(name);
	}
	return result;
}
#endif

ExecuteContext::ExecuteContext() {
	mutex.lock();
}

ExecuteContext::~ExecuteContext() {
	mutex.unlock();
}

void ExecuteContext::set_var(const std::string& var_name, const std::string& var_value, bool global) {
	std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
	if (!lock.try_lock()) {
		throw std::runtime_error("Variables can be set only during 'exec' action");
	}
	vars[var_name] = var_value;
	if (global) {
		global_vars.insert(var_name);
	}
}

std::string ExecuteContext::get_var(const std::string& var_name) {
	std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
	if (!lock.try_lock()) {
		throw std::runtime_error("Variables can be get only during 'exec' action");
	}
	auto it = vars.find(var_name);
	if (it == vars.end()) {
		throw std::runtime_error("Can't find a variable with a name " + var_name);
	}
	return it->second;
}

void ExecuteContext::unlock(const nlohmann::json& j) {
	vars.clear();
	global_vars.clear();

	for (auto& item: j.items()) {
		vars[item.key()] = item.value();
	}

	mutex.unlock();
}

nlohmann::json ExecuteContext::lock() {
	mutex.lock();

	nlohmann::json j = nlohmann::json::array();
	for (auto& kv: vars) {
		j.push_back({
			{"name", kv.first},
			{"value", kv.second},
			{"global", (bool)global_vars.count(kv.first)},
		});
	}
	return j;
}

void MessageHandler::run(std::shared_ptr<Channel> channel_) {
	spdlog::info("Waiting for commands");

	channel = channel_;
	SCOPE_EXIT { channel = {}; };

	while (true) {
		command = channel->receive();
		SCOPE_EXIT { command = {}; };

		handle_message();
	}
}

void MessageHandler::handle_message() {
	if (command.count("version")) {
		ver = command.at("version").get<std::string>();
	} else {
		ver = VersionNumber();
	}

	spdlog::info("testo version = {}, testo guest additions version = {}", ver.to_string(), TESTO_VERSION);

	try {
		do_handle_message(command.at("method").get<std::string>());
	} catch (const std::exception& error) {
		spdlog::error("Error in MessageHandler::handle_message: {}", error.what());
#ifdef WIN32
		if (dynamic_cast<const std::system_error*>(&error) && !dynamic_cast<const fs::filesystem_error*>(&error)) {
			std::wstring utf16_err = winapi::acp_to_utf16(error.what());
			std::string utf8_err = winapi::utf16_to_utf8(utf16_err);
			send_error(utf8_err);
		} else {
			send_error(error.what());
		}
#else
		send_error(error.what());
#endif
	}
}

void MessageHandler::do_handle_message(const std::string& method_name) {
	if (method_name == "check_avaliable") {
		return handle_check_avaliable();
	} else if (method_name == "get_tmp_dir") {
		return handle_get_tmp_dir();
	} else if (method_name == "copy_file") {
		return handle_copy_file();
	} else if (method_name == "copy_files_out") {
		return handle_copy_files_out();
	} else if (method_name == "execute") {
		return handle_execute();
	} else if (method_name == "mount") {
		return handle_mount();
	} else if (method_name == "get_shared_folder_status") {
		return handle_get_shared_folder_status();
	} else if (method_name == "umount") {
		return handle_umount();
	} else {
		throw std::runtime_error(std::string("Method ") + method_name + " is not supported");
	}
}

void MessageHandler::send_error(const std::string& error) {
	nlohmann::json response = {
		{"success", false},
	};

	if (ver < VersionNumber(2,2,10)) {
		response["error"] = error;
	} else {
		response["error"] = base64_encode((uint8_t*)error.data(), error.size() + 1);
	}

	channel->send(std::move(response));
}

void MessageHandler::handle_check_avaliable() {
	spdlog::info("Checking avaliability call");

	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	channel->send(std::move(response));
	spdlog::info("Checking avaliability is OK");
}

void MessageHandler::handle_get_tmp_dir() {
	spdlog::info("Getting tmp dir");

	nlohmann::json response = {
		{"success", true},
		{"result", {
			{"path", fs::temp_directory_path().generic_string()}
		}}
	};

	channel->send(std::move(response));
	spdlog::info("Getting tmp dir is OK");
}

void MessageHandler::handle_copy_file() {
	const nlohmann::json& args = command.at("args");

	for (auto file: args) {
		fs::path dst = file.at("path").get<std::string>();
		spdlog::info("Copying file to guest: " + dst.generic_string());

		if (dst.is_relative()) {
			throw std::runtime_error("Destination path on vm must be absolute");
		}

		if (!fs::exists(dst.parent_path())) {
			if (!fs::create_directories(dst.parent_path())) {
				throw std::runtime_error("Can't create directory: " + dst.parent_path().generic_string());
			}
		}

		os::File f = os::File::open_for_write(dst.generic_string());
		if (file.at("content").is_string()) {
			auto content64 = file.at("content").get<std::string>();
			auto content = base64_decode(content64);
			f.write(content.data(), content.size());
		} else {
			uint64_t file_length = 0;
			channel->receive_raw((uint8_t*)&file_length, sizeof(file_length));
			uint64_t i = 0;
			const uint64_t buf_size = 8 * 1024;
			uint8_t buf[buf_size];
			while (i < file_length) {
				uint64_t chunk_size = std::min(buf_size, file_length - i);
				channel->receive_raw(buf, chunk_size);
				f.write(buf, chunk_size);
				i += chunk_size;
			}
		}

		spdlog::info("File copied successfully to guest: " + dst.generic_string());
	}

	nlohmann::json response = {
		{"success", true},
		{"result", nlohmann::json::object()}
	};

	channel->send(std::move(response));
}

nlohmann::json MessageHandler::copy_single_file_out(const fs::path& src, const fs::path& dst) {
	if (src.is_relative()) {
		throw std::runtime_error(fmt::format("Source path on vm must be absolute"));
	}

	nlohmann::json result = {
		{"src", src.generic_string()},
		{"path", dst.generic_string()},
	};

	os::File file = os::File::open_for_read(src.generic_string());
	if (ver < VersionNumber(2,2,8)) {
		std::vector<uint8_t> fileContents = file.read_all();
		std::string encoded = base64_encode(fileContents.data(), (uint32_t)fileContents.size());
		result["content"] = std::move(encoded);
	} else {
		result["content"] = nullptr;
		result["content_size"] = file.size();
	}

	return result;
}

nlohmann::json MessageHandler::copy_directory_out(const fs::path& dir, const fs::path& dst) {
	nlohmann::json files = nlohmann::json::array();

	files.push_back({
		{"path", dst.generic_string()}
	});

	for (auto& file: fs::directory_iterator(dir)) {
		if (fs::is_regular_file(file)) {
			files.push_back(copy_single_file_out(file, dst / fs::path(file).filename()));
		} else if (fs::is_directory(file)) {
			auto result = copy_directory_out(file, dst / fs::path(file).filename());
			files.insert(files.end(), result.begin(), result.end());
		} else {
			throw std::runtime_error("Unknown type of file: " + fs::path(file).generic_string());
		}
	}

	return files;
}

void MessageHandler::handle_copy_files_out() {
	const nlohmann::json& args = command.at("args");

	nlohmann::json files = nlohmann::json::array();
	fs::path src = args[0].get<std::string>();
	fs::path dst = args[1].get<std::string>();

	spdlog::info("Copying FROM guest: " + src.generic_string());

	if (!fs::exists(src)) {
		throw std::runtime_error("Source " + src.generic_string() + " doesn't exist on guest");
	}

	if (fs::is_regular_file(src)) {
		files.push_back(copy_single_file_out(src, dst));
	} else if (fs::is_directory(src)) {
		auto result = copy_directory_out(src, dst);
		files.insert(files.end(), result.begin(), result.end());
	} else {
		throw std::runtime_error("Unknown type of file: " + src.generic_string());
	}

	nlohmann::json result = {
		{"success", true},
		{"result", files}
	};

	channel->send(std::move(result));

	if (ver < VersionNumber(2,2,8)) {
		// do nothing
	} else {
		for (auto& item: files) {
			if (!item.count("content")) {
				continue;
			}
			std::string path = item.at("src");
			os::File file = os::File::open_for_read(path);
			uint64_t file_length = file.size();
			spdlog::info("Sending file {}, file size = {}", path, file_length);
			channel->send_raw((uint8_t*)&file_length, sizeof(file_length));
			uint64_t i = 0;
			const uint64_t buf_size = 8 * 1024;
			uint8_t buf[buf_size];
			while (i < file_length) {
				uint64_t chunk_size = std::min(buf_size, file_length - i);
				file.read(buf, chunk_size);
				channel->send_raw(buf, chunk_size);
				i += chunk_size;
			}
		}
	}

	spdlog::info("Copied FROM guest: " + src.generic_string());
}

void MessageHandler::handle_execute() {
	const nlohmann::json& args = command.at("args");
	auto cmd = args[0].get<std::string>();

	spdlog::info("Executing command " + cmd);

	int rc = 0;
	nlohmann::json vars;
	if (command.count("vars")) {
		vars = command.at("vars");
	} else {
		vars = nlohmann::json::object();
	}

	{
		exec_ctx.unlock(vars);
		SCOPE_EXIT { vars = exec_ctx.lock(); };
		rc = do_handle_execute(cmd);
	}

	nlohmann::json result = {
		{"success", true},
		{"result", {
			{"status", "finished"},
			{"exit_code", rc},
			{"vars", vars},
		}}
	};

	channel->send(std::move(result));

	spdlog::info("Command finished: " + cmd);
	spdlog::info("Return code: " + std::to_string(rc));
}

int MessageHandler::do_handle_execute(std::string cmd) {
#if __linux__
	cmd += " 2>&1";
#endif

#if WIN32
	std::map<std::string, std::string> env_vars = winapi::get_environment_strings();
	for (auto& kv: get_environment_from_registry()) {
		env_vars[kv.first] = kv.second;
	}
	os::Process process(cmd, &env_vars);
#else
	os::Process process(cmd);
#endif

	while (!process.eof()) {
		std::string output = process.read();
		if (output.size()) {
			nlohmann::json result = {
				{"success", true},
				{"result", {
					{"status", "pending"},
					{"stdout", base64_encode((uint8_t*)output.data(), output.size() + 1)}
				}}
			};
			channel->send(std::move(result));
		}
	}

	return process.wait();
}

void MessageHandler::handle_mount() {
	const nlohmann::json& args = command.at("args");
	std::string folder_name = args.at("folder_name");
	fs::path guest_path = args.at("guest_path").get<std::string>();
	bool permanent = args.at("permanent");

	spdlog::info("Mounting shared folder {} to {}", folder_name, guest_path.generic_string());

	bool was_indeed_mounted = mount_shared_folder(folder_name, guest_path);

	if (permanent) {
		register_shared_folder(folder_name, guest_path);
	}

	nlohmann::json result = {
		{"success", true},
		{"was_indeed_mounted", was_indeed_mounted}
	};

	channel->send(std::move(result));

	spdlog::info("Mounting is OK");
}

void MessageHandler::handle_get_shared_folder_status() {
	const nlohmann::json& args = command.at("args");
	std::string folder_name = args.at("folder_name");

	spdlog::info("Getting shared folder {} status", folder_name);

	nlohmann::json status = get_shared_folder_status(folder_name);

	nlohmann::json result = {
		{"success", true},
		{"result", status}
	};

	channel->send(std::move(result));

	spdlog::info("Getting status is OK");
}

void MessageHandler::handle_umount() {
	const nlohmann::json& args = command.at("args");
	std::string folder_name = args.at("folder_name");
	bool permanent = args.at("permanent");

	spdlog::info("Umounting shared folder {}", folder_name);

	bool was_indeed_umounted = umount_shared_folder(folder_name);

	if (permanent) {
		unregister_shared_folder(folder_name);
	}

	nlohmann::json result = {
		{"success", true},
		{"was_indeed_umounted", was_indeed_umounted}
	};

	channel->send(std::move(result));

	spdlog::info("Umounting is OK");
}

void CLIMessageHandler::do_handle_message(const std::string& method_name) {
	if (method_name == "set_var") {
		return handle_set_var();
	} else if (method_name == "get_var") {
		return handle_get_var();
	} else {
		return MessageHandler::do_handle_message(method_name);
	}
}

void CLIMessageHandler::handle_set_var() {
	const nlohmann::json& args = command.at("args");
	std::string var_name = args.at("var_name");
	std::string var_value = args.at("var_value");
	bool global = args.at("global");

	spdlog::info("Setting variable {} = {}", var_name, var_value);

	host_handler->exec_ctx.set_var(var_name, var_value, global);

	nlohmann::json result = {
		{"success", true},
	};

	channel->send(std::move(result));

	spdlog::info("Setting variable is OK");
}

void CLIMessageHandler::handle_get_var() {
	const nlohmann::json& args = command.at("args");
	std::string var_name = args.at("var_name");

	spdlog::info("Getting variable {}", var_name);

	std::string var_value = host_handler->exec_ctx.get_var(var_name);

	nlohmann::json result = {
		{"success", true},
		{"var_value", var_value}
	};

	channel->send(std::move(result));

	spdlog::info("Getting variable is OK");
}
