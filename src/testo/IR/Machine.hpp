
#pragma once

#include "Controller.hpp"
#include "../backends/VM.hpp"
#include <unordered_map>

namespace IR {

struct Machine: Controller {
	static std::string type_name() { return "virtual machine"; }

	std::shared_ptr<::VM> vm() const;

	virtual std::string type() const override;

	virtual void create_snapshot(const std::string& snapshot, const std::string& cksum, bool hypervisor_snapshot_needed) override;
	virtual void restore_snapshot(const std::string& snapshot) override;
	virtual void delete_snapshot_with_children(const std::string& snapshot) override;

	virtual bool check_config_relevance() override;

	virtual bool is_defined() const override;
	virtual void create() override;
	virtual void undefine() override;

	bool is_nic_plugged(const std::string& nic);

	void plug_nic(const std::string& nic);
	void unplug_nic(const std::string& nic);

	bool is_link_plugged(const std::string& nic);

	void plug_link(const std::string& nic);
	void unplug_link(const std::string& nic);

	void hold(KeyboardButton button);
	void release();
	void release(KeyboardButton button);

	void mouse_hold(MouseButton button);
	void mouse_release();

	void validate_config();

	const stb::Image<stb::RGB>& make_new_screenshot();
	const stb::Image<stb::RGB>& get_last_screenshot() const;

	const std::map<std::string, std::string>& get_vars() const;
	void set_var(const std::string& var_name, const std::string& var_value);

private:
	stb::Image<stb::RGB> _last_screenshot;

	virtual std::string id() const override;
	virtual fs::path get_metadata_dir() const override;

	mutable std::shared_ptr<::VM> _vm;

	MouseButton current_held_mouse_button = MouseButton::None;
	std::vector<KeyboardButton> current_held_keyboard_buttons;
	std::map<std::string, std::string> vars;
};

}
