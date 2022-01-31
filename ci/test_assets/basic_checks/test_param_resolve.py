from common import *

def test_for_range_param():
	out, err = must_succeed("testo run param_resolve/for_tests.testo --test_spec for_range_param --invalidate for_range_param")
	assert "0: iteration #0" in out
	assert "0: iteration #1" in out
	assert "0: iteration #2" in out
	assert "0: iteration #3" in out

	assert "1: iteration #0" in out
	assert "1: iteration #1" in out
	assert "1: iteration #2" in out
	assert "1: iteration #3" not in out


	out, err = must_succeed("testo run param_resolve/for_tests.testo --test_spec for_range_str_0 --invalidate for_range_str_0")
	assert "iteration #3" in out
	assert "iteration #4" in out
	assert "iteration #5" not in out


	out, err = must_succeed("testo run param_resolve/for_tests.testo --test_spec for_range_token --invalidate for_range_token")
	assert "iteration #3" in out
	assert "iteration #4" in out
	assert "iteration #5" not in out


	out, err = must_succeed("testo run param_resolve/for_tests.testo --test_spec for_copyto --invalidate for_copyto")
	assert f"Copying {cwd}/param_resolve/./0000.txt to flash drive my_flash to destination /0000.txt" in out
	assert f"Copying {cwd}/param_resolve/./0001.txt to flash drive my_flash to destination /0001.txt" in out
	assert f"Copying {cwd}/param_resolve/./0002.txt to flash drive my_flash to destination /0002.txt" in out
	assert f"Copying {cwd}/param_resolve/./0003.txt to flash drive my_flash to destination /0003.txt" in out
	assert f"Copying {cwd}/param_resolve/./0004.txt to flash drive my_flash to destination /0004.txt" in out
	assert f"Copying {cwd}/param_resolve/./0005.txt to flash drive my_flash to destination /0005.txt" not in out

def test_actions_param():
	must_succeed("testo run param_resolve/actions.testo --test_spec check_sleep_param --invalidate check_sleep_param", 'Sleeping in virtual machine my_ubuntu_server for 1s')
	must_succeed("testo run param_resolve/actions.testo --test_spec check_sleep_token --invalidate check_sleep_token", 'Sleeping in virtual machine my_ubuntu_server for 1s')

	must_fail("testo run param_resolve/actions.testo --test_spec check_type_param --invalidate check_type_param", out='Typing "Hello 1s" with interval 1s in virtual machine my_ubuntu_server')
	must_fail("testo run param_resolve/actions.testo --test_spec check_type_token --invalidate check_type_token", out='Typing "Hello 1s" with interval 1s in virtual machine my_ubuntu_server')

	must_fail("testo run param_resolve/actions.testo --test_spec check_wait_param --invalidate check_wait_param", out='Waiting "Hello 200ms" for 200ms with interval 200ms in virtual machine my_ubuntu_server')
	must_fail("testo run param_resolve/actions.testo --test_spec check_wait_token --invalidate check_wait_token", out='Waiting "Hello 200ms" for 200ms with interval 200ms in virtual machine my_ubuntu_server')

	must_fail("testo run param_resolve/actions.testo --test_spec check_shutdown_param --invalidate check_shutdown_param", out='Shutting down virtual machine my_ubuntu_server with timeout 200ms')
	must_fail("testo run param_resolve/actions.testo --test_spec check_shutdown_token --invalidate check_shutdown_token", out='Shutting down virtual machine my_ubuntu_server with timeout 200ms')

	must_fail("testo run param_resolve/actions.testo --test_spec check_copyto_param --invalidate check_copyto_param", out=f'Copying {cwd}/param_resolve/actions.testo to virtual machine my_ubuntu_server to destination /opt/scripts/actions.testo with timeout 200ms')
	must_fail("testo run param_resolve/actions.testo --test_spec check_copyto_token --invalidate check_copyto_token", out=f'Copying {cwd}/param_resolve/actions.testo to virtual machine my_ubuntu_server to destination /opt/scripts/actions.testo with timeout 200ms')

	must_fail("testo run param_resolve/actions.testo --test_spec check_exec_param --invalidate check_exec_param", out='Executing bash command in virtual machine my_ubuntu_server with timeout 200ms')
	must_fail("testo run param_resolve/actions.testo --test_spec check_exec_token --invalidate check_exec_token", out='Executing bash command in virtual machine my_ubuntu_server with timeout 200ms')

	must_succeed("testo run param_resolve/actions.testo --test_spec check_check_param --invalidate check_check_param", 'Checking "Hello 200ms" for 200ms with interval 300ms in virtual machine my_ubuntu_server')
	must_succeed("testo run param_resolve/actions.testo --test_spec check_check_token --invalidate check_check_token", 'Checking "Hello 200ms" for 200ms with interval 300ms in virtual machine my_ubuntu_server')

	must_fail("testo run param_resolve/actions.testo --test_spec check_press_param --invalidate check_press_param", out='Pressing key A 2 times in virtual machine my_ubuntu_server')
	must_fail("testo run param_resolve/actions.testo --test_spec check_press_token --invalidate check_press_token", out='Pressing key A 2 times in virtual machine my_ubuntu_server')

	must_succeed("testo run param_resolve/actions.testo --test_spec check_flash_param --invalidate check_flash_param", out='Plugging flash drive my_flash into virtual machine my_ubuntu_server')
	must_fail("testo run param_resolve/actions.testo --test_spec check_flash_token --invalidate check_flash_token", out='Unplugging flash drive my_flash from virtual machine my_ubuntu_server')

	must_fail("testo run param_resolve/actions.testo --test_spec check_nic_param --invalidate check_nic_param", out='Plugging nic my_nic into virtual machine my_ubuntu_server')
	must_succeed("testo run param_resolve/actions.testo --test_spec check_nic_token --invalidate check_nic_token", out='Unplugging nic my_nic from virtual machine my_ubuntu_server')

	must_fail("testo run param_resolve/actions.testo --test_spec check_link_param --invalidate check_link_param", out='Plugging link my_nic into virtual machine my_ubuntu_server')
	must_succeed("testo run param_resolve/actions.testo --test_spec check_link_token --invalidate check_link_token", out='Unplugging link my_nic from virtual machine my_ubuntu_server')

	must_fail("testo run param_resolve/actions.testo --test_spec check_mouse_param --invalidate check_mouse_param", out='Mouse moving on "lala" with timeout 200ms in virtual machine my_ubuntu_server')
	must_fail("testo run param_resolve/actions.testo --test_spec check_mouse_token --invalidate check_mouse_token", out='Mouse moving on "lala" with timeout 200ms in virtual machine my_ubuntu_server')

	must_fail("testo run param_resolve/actions.testo --test_spec check_sleep_param_invalid", f"""{cwd}/param_resolve/actions.testo:30:9: Error while parsing "-7"
	- Error: expected TIME INTERVAL, but got NUMBER "-7\"""")

	must_fail("testo run param_resolve/actions.testo --test_spec check_type_param_invalid", f"""{cwd}/param_resolve/actions.testo:53:39: Error while parsing "interval"
	- Error: expected TIME INTERVAL, but got IDENTIFIER "interval\"""")

	must_fail("testo run param_resolve/actions.testo --test_spec check_wait_param_invalid", f"""{cwd}/param_resolve/actions.testo:77:39: Error while parsing "interval"
	- Error: expected TIME INTERVAL, but got IDENTIFIER "interval\"""")

	must_fail("testo run param_resolve/actions.testo --test_spec check_shutdown_param_invalid", f"""{cwd}/param_resolve/actions.testo:100:20: Error while parsing ""
	- Error: expected TIME INTERVAL, but got EOF""")

def test_param_not_defined():
	must_fail("testo run param_resolve/not_defined.testo --test_spec sleep_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec print_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec abort_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec type_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec wait_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec mouse_text_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec mouse_js_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec flash_not_defined", f"""In a macro call plug_flash("undefined_flash")
{cwd}/param_resolve/not_defined.testo:73:7: Error: unknown flash drive: undefined_flash""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec dvd_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec from_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec to_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec exec_not_defined", """Error while resolving "${not_defined}"
	- param "not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec macro_arg_not_defined", """Error while resolving "${arg_not_defined}"
	- param "arg_not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec macro_default_arg_not_defined", """Error while resolving "${default_not_defined}"
	- param "default_not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec factor_not_defined", """Error while resolving "${factor_not_defined}"
	- param "factor_not_defined" is not defined""")

	must_fail("testo run param_resolve/not_defined.testo --test_spec attr_not_defined", """Error while resolving "${iso_not_defined}"
	- param "iso_not_defined" is not defined""")

def test_cmd_param():
	must_succeed("testo run param_resolve/commands.testo --test_spec machine_as_string --invalidate machine_as_string", "my_ubuntu_server: Hello world")
	must_succeed("testo run param_resolve/commands.testo --test_spec flash_as_string --invalidate flash_as_string", "my_flash: Hello world")

def test_cmd_macro_args():
	must_succeed("testo run param_resolve/commands.testo --test_spec basic_macro_command --invalidate basic_macro_command", f"""[  0%] Calling command macro some_command_macro(vm1="my_ubuntu_server", string1="Hello world")
[  0%] Starting virtual machine my_ubuntu_server
[  0%] Typing "Hello world" with interval 30ms in virtual machine my_ubuntu_server
""")

	must_succeed("testo run param_resolve/commands.testo --test_spec basic_macro_command_flash --invalidate basic_macro_command_flash", f"""[  0%] Calling command macro some_command_flash_macro(fd1="my_flash", string1="Hello world")
[  0%] my_flash: Hello world
""")

	must_succeed("testo run param_resolve/commands.testo --test_spec check_empty_macros --invalidate check_empty_macros", f"""[  0%] Calling macro empty_macro1() in virtual machine my_ubuntu_server
[  0%] Calling macro empty_macro2() in virtual machine my_ubuntu_server
[  0%] Calling macro empty_macro1() in virtual machine my_ubuntu_server
[  0%] Calling macro empty_macro2() in virtual machine my_ubuntu_server
""")
