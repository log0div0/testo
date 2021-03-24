
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

	void press(const std::vector<std::string>& buttons);
	void hold(const std::vector<std::string>& buttons);
	void release();
	void release(const std::vector<std::string>& buttons);

	void mouse_press(const std::vector<MouseButton>& buttons);
	void mouse_hold(const std::vector<MouseButton>& buttons);
	void mouse_release();

	void validate_config();

private:
	virtual std::string id() const override;
	virtual fs::path get_metadata_dir() const override;

	mutable std::shared_ptr<::VM> _vm;

	MouseButton current_held_mouse_button = MouseButton::None;
	std::vector<std::string> current_held_keyboard_buttons;
};

}
