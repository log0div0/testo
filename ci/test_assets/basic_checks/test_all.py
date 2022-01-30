
from subprocess import Popen, PIPE
from sys import platform
import os, re

cwd = os.getcwd().replace('\\', '/')

if 'HYPERVISOR' in os.environ:
	HYPERVISOR = os.environ['HYPERVISOR']
else:
	if platform == "linux":
		HYPERVISOR = "qemu"
	elif platform == "win32":
		HYPERVISOR = "hyperv"
	else:
		raise "Please, specify hypervisor"

def must_succeed(cmd, exit_code = None):
	p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
	out, err = p.communicate()
	out, err = out.decode('utf-8'), err.decode('utf-8')
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", cmd, " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
	print("STDOUT:", out)
	print("STDERR:", err)
	assert p.returncode == 0
	return out, err

def must_fail(cmd, x = None):
	p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
	out, err = p.communicate()
	out, err = out.decode('utf-8'), err.decode('utf-8')
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", cmd, " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
	print("STDOUT:", out)
	print("STDERR:", err)
	if x is None:
		assert p.returncode != 0
	elif isinstance(x, str):
		assert p.returncode != 0
		assert x in err
	else:
		assert p.returncode == x
	return out, err

def test_version():
	must_succeed("testo version")

def test_parser():
	must_fail("testo run parser/macro_body_eof.testo", 						'Error: macro "some_macro" body reached the end of file without closing "}"')
	must_fail("testo run parser/invalid_mouse_parented_expr_0.testo", 		'Error: Unknown selective object type: (')
	must_fail("testo run parser/invalid_mouse_parented_expr_1.testo", 		'Error: Unknown selective object type: !')
	must_fail("testo run parser/vm_identical_disks.testo", 					'Error: duplicate attribute: "disk first"')
	must_fail("testo run parser/vm_identical_nic_names.testo", 				'Error: duplicate attribute: "nic my_nic"')
	must_fail("testo run parser/vm_unknown_attr.testo", 					'Error: Unknown attribute: some_attr')
	must_fail("testo run parser/vm_attr_requires_name.testo", 				'Error: unexpected token :, expected: IDENTIFIER')
	must_fail("testo run parser/vm_attr_must_have_no_name.testo", 			'Error: unexpected token IDENTIFIER, expected: :')
	must_fail("testo run parser/vm_attr_type_mismatch.testo", 				'Error: expected STRING or SIZE, but got NUMBER "4"')
	must_fail("testo run parser/vm_attr_diplucates.testo", 					'Error: duplicate attribute: "cpus"')

def test_semantic():
	must_fail("testo run semantic/0001.testo", f"""{cwd}/semantic/0001.testo:6:1: Error: test "A" is already defined

{cwd}/semantic/0001.testo:2:1: note: previous declaration was here""")

	must_fail("testo run semantic/0002.testo", f"""{cwd}/semantic/0002.testo:6:1: Error: macro "A" is already defined

{cwd}/semantic/0002.testo:2:1: note: previous declaration was here""")

	must_fail("testo run semantic/0003.testo", f"""{cwd}/semantic/0003.testo:4:1: Error: param "A" is already defined

{cwd}/semantic/0003.testo:2:1: note: previous declaration was here""")

	must_fail("testo run semantic/0004.testo", f"""{cwd}/semantic/0004.testo:6:1: Error: virtual machine "A" is already defined

{cwd}/semantic/0004.testo:2:1: note: previous declaration was here""")

	must_fail("testo run semantic/0005.testo", f"""{cwd}/semantic/0005.testo:6:1: Error: flash drive "A" is already defined

{cwd}/semantic/0005.testo:2:1: note: previous declaration was here""")

	must_fail("testo run semantic/0006.testo", f"""{cwd}/semantic/0006.testo:6:1: Error: network "A" is already defined

{cwd}/semantic/0006.testo:2:1: note: previous declaration was here""")


	must_fail("testo run semantic/0007.testo", "/0007.testo:2:9: Error: can't specify test as a parent to itself A")
	must_fail("testo run semantic/0008.testo", '/0008.testo:2:9: Error: unknown test: B')
	must_fail("testo run semantic/0009.testo", '/0009.testo:6:12: Error: this test was already specified in parent list A')
	must_fail("testo run semantic/0010.testo", f'/0010.testo:6:1: Error: virtual machine "A" is already defined here: {cwd}/semantic/0010.testo:2:1')


	must_fail("testo run semantic/macro_call.testo --test_spec unknown_command_macro", 					"Error: unknown macro: some_macro")
	must_fail("testo run semantic/macro_call.testo --test_spec too_few_args_command", 					"Error: expected at least 2 args, 1 provided")
	must_fail("testo run semantic/macro_call.testo --test_spec too_many_args_command", 					"Error: expected at most 4 args, 5 provided")
	must_fail("testo run semantic/macro_call.testo --test_spec wrong_macro_0_command", 					"Error: duplicate macro arg: arg1")
	must_fail("testo run semantic/macro_call.testo --test_spec wrong_macro_1_command", 					"Error: default value must be specified for macro arg arg3")
	must_fail("testo run semantic/macro_call.testo --test_spec call_macro_action_outside_command", 		"Error: unexpected token ACTION PRINT, expected: }")
	must_fail("testo run semantic/macro_call.testo --test_spec call_macro_action_outside_command2", 	'Error: the "my_macro" macro does not contain commands, as expected')
	must_fail("testo run semantic/macro_call.testo --test_spec call_macro_command_inside_command", 		"Error: Unknown action: my_ubuntu_server")
	must_fail("testo run semantic/macro_call.testo --test_spec call_macro_command_inside_command2", 	'Error: the "my_command_macro" macro does not contain actions, as expected')
	must_fail("testo run semantic/macro_call.testo --test_spec call_nested_macro1", 					"Error: Unknown action: my_ubuntu_server")
	must_fail("testo run semantic/macro_call.testo --test_spec call_nested_macro2", 					"Error: unexpected token ACTION PRINT, expected: }")
	must_fail("testo run semantic/macro_call.testo --test_spec call_macro_starts_with_if", 				"Error: unexpected token IF, expected: }")
	must_fail("testo run semantic/macro_call.testo --test_spec call_macro_starts_with_for", 			"Error: unexpected token FOR, expected: }")
	must_fail("testo run semantic/macro_call.testo --test_spec call_macro_starts_with_semicolon", 		"Error: unexpected token ;, expected: }")
	must_fail("testo run semantic/macro_call.testo --test_spec wrong_flash_macro2", 					'Error: The action "type "lala"" is not applicable to a flash drive')
	must_fail("testo run semantic/macro_call.testo --test_spec call_macro_test_outside_test1", 			"Error: unexpected token TEST, expected: }")


	out, err = must_succeed('testo run semantic/macro_call.testo --test_spec call_macro_inside_if_check1 --invalidate call_macro_inside_if_check1')
	assert "Hello world1" in out
	assert "Hello world2" not in out

	out, err = must_succeed('testo run semantic/macro_call.testo --test_spec call_macro_inside_if_check2 --invalidate call_macro_inside_if_check2')
	assert "Hello world1" not in out
	assert "Hello world2" in out


	must_fail("testo run semantic/expr.testo --test_spec comparison_0", 'Error: "lala" is not an integer number')
	must_fail("testo run semantic/expr.testo --test_spec comparison_1", 'Error: "lala" is not an integer number')
	must_fail("testo run semantic/expr.testo --test_spec comparison_2", 'Error: "30ms" is not an integer number')
	must_fail("testo run semantic/expr.testo --test_spec comparison_3", 'Error: "-5ms" is not an integer number')
	must_fail("testo run semantic/expr.testo --test_spec comparison_4", 'Error: "Русский" is not an integer number')
	must_fail("testo run semantic/expr.testo --test_spec comparison_5", 'Error: "- 5ms" is not an integer number')
	must_fail("testo run semantic/expr.testo --test_spec check_flash", 'Error: The "check" expression is not applicable to a flash drive')


	must_fail("testo run semantic/misc.testo --test_spec empty_wait_text", 				'Error: empty string in text selection')
	must_fail("testo run semantic/misc.testo --test_spec empty_mouse_move_text", 		'Error: empty string in text selection')
	must_fail("testo run semantic/misc.testo --test_spec empty_wait_js", 				'Error: empty script in js selection')
	must_fail("testo run semantic/misc.testo --test_spec empty_mouse_move_js", 			'Error: empty script in js selection')
	must_fail("testo run semantic/misc.testo --test_spec invalid_wait_js", 				'Error while validating js selection')
	must_fail("testo run semantic/misc.testo --test_spec invalid_mouse_move", 			'Error while validating js selection')
	must_fail("testo run semantic/misc.testo --test_spec invalid_key_0", 				'Error: unknown key: SomeButton')
	must_fail("testo run semantic/misc.testo --test_spec invalid_key_1", 				"Error: duplicate key: LeftShift")
	must_fail("testo run semantic/misc.testo --test_spec invalid_press_times_0", 		"Error: can't press a button less than 1 time: 0")
	must_fail("testo run semantic/misc.testo --test_spec invalid_press_times_1", 		"Error: can't press a button less than 1 time: -1")
	must_fail("testo run semantic/misc.testo --test_spec invalid_press_times_2", 		"Error: can't press a button less than 1 time: -1")
	must_fail("testo run semantic/misc.testo --test_spec unknown_flash_0", 				"Error: unknown flash drive: unexisting_flash")
	must_fail("testo run semantic/misc.testo --test_spec unknown_flash_1", 				"Error: unknown flash drive: unexisting_flash")
	must_fail("testo run semantic/misc.testo --test_spec unknown_vm", 					"Error: unknown virtual entity: some_vm")
	must_fail("testo run semantic/misc.testo --test_spec common_vm_child", 				"Error: some parents have common virtual machines")
	must_fail("testo run semantic/misc.testo --test_spec common_flash_child", 			"Error: some parents have common flash drives")
	must_fail("testo run semantic/misc.testo --test_spec copyfrom_nocheck", 			'Error: "nocheck" specifier is not applicable to copyfrom action')
	must_fail("testo run semantic/misc.testo --test_spec copyto_from_not_exist", 		"Error: specified path doesn't exist: /some/unexisting_shit")
	must_fail("testo run semantic/misc.testo --test_spec flash_copyto_from_not_exist", 	"Error: specified path doesn't exist: /some/unexisting_shit")
	must_fail("testo run semantic/misc.testo --test_spec img_doesnt_exist1", 			f"Error: specified image path does not exist: {cwd}/semantic/someshit")
	must_fail("testo run semantic/misc.testo --test_spec img_doesnt_exist2", 			f"Error: specified image path does not exist: {cwd}/semantic/./someshit")
	must_fail("testo run semantic/misc.testo --test_spec img_doesnt_lead_to_a_file", 	"Error: specified image path does not lead to a regular file: /usr/sbin")
	must_fail("testo run semantic/misc.testo --test_spec invalid_hostdev_usb_addr1", 	"Error: spicified usb addr is not valid")
	must_fail("testo run semantic/misc.testo --test_spec invalid_hostdev_usb_addr2", 	"Error: spicified usb addr is not valid")
	must_fail("testo run semantic/misc.testo --test_spec invalid_hostdev_usb_addr3", 	"Error: spicified usb addr is not valid")
	must_fail("testo run semantic/misc.testo --test_spec invalid_hostdev_usb_addr4", 	"Error: spicified usb addr is not valid")
	must_fail("testo run semantic/misc.testo --test_spec invalid_hostdev_usb_addr5", 	"Error: spicified usb addr is not valid")
	must_fail("testo run semantic/misc.testo --test_spec invalid_hostdev_usb_addr6", 	"Error: spicified usb addr is not valid")


	must_fail("testo run semantic/mouse_specificators.testo --test_spec from_0", 					"""Error: specifier from_top requires a non-negative number as an argument""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec from_1", 					"""Error: specifier from_left requires a non-negative number as an argument""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec from_2", 					"""Error: you can't use specifier from_top after another "from" specifier""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec from_3", 					"""Error: you can't use specifier from_top after a "precision" specifier""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec from_4", 					"""Error: you can't use specifier from_left after a "move" specifier""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec centering_0", 				"""Error: specifier center must not have an argument""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec centering_1", 				"""Error: you can't use specifier right_top after another "precision" specifier""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec centering_2", 				"""Error: you can't use specifier right_center after a "move" specifier""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec move_0", 					"""Error: specifier move_right requires a non-negative number as an argument""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec move_1", 					"""Error: specifier move_right requires a non-negative number as an argument""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec unknown", 					"""Error: unknown specifier: move_bottom""")
	must_fail("testo run semantic/mouse_specificators.testo --test_spec js_with_specificators", 	"""Error: mouse specifiers are not supported for js selections""")


	must_fail("testo run semantic/macro_call.testo --test_spec unknown_macro", 		'Error: unknown macro: some_macro')
	must_fail("testo run semantic/macro_call.testo --test_spec too_few_args", 		'Error: expected at least 2 args, 1 provided')
	must_fail("testo run semantic/macro_call.testo --test_spec too_many_args", 		'Error: expected at most 4 args, 5 provided')
	must_fail("testo run semantic/macro_call.testo --test_spec wrong_macro_0", 		'Error: duplicate macro arg: arg1')
	must_fail("testo run semantic/macro_call.testo --test_spec wrong_macro_1", 		'Error: default value must be specified for macro arg arg3')
	must_fail("testo run semantic/macro_call.testo --test_spec wrong_flash_macro", 	'Error: The action "type "lala"" is not applicable to a flash drive')


	must_fail("testo run semantic/for.testo --test_spec wrong_r1_0", 			'Error: expected NUMBER, but got TIME INTERVAL "30ms"')
	must_fail("testo run semantic/for.testo --test_spec wrong_r1_1", 			"Can't convert range start -5 to a non-negative number")
	must_fail("testo run semantic/for.testo --test_spec wrong_r1_2", 			"Can't convert range start -5 to a non-negative number")
	must_fail("testo run semantic/for.testo --test_spec wrong_r2_0", 			'Error: expected NUMBER, but got TIME INTERVAL "30ms"')
	must_fail("testo run semantic/for.testo --test_spec wrong_r2_1", 			"Can't convert range finish -5 to a non-negative number")
	must_fail("testo run semantic/for.testo --test_spec wrong_r2_2", 			'Error: expected NUMBER, but got TIME INTERVAL "30ms"')
	must_fail("testo run semantic/for.testo --test_spec wrong_r2_3", 			"Can't convert range finish -5 to a non-negative number")
	must_fail("testo run semantic/for.testo --test_spec wrong_r2_4", 			"Can't convert range finish -5 to a non-negative number")
	must_fail("testo run semantic/for.testo --test_spec wrong_start_finish_0", 	'Error: start of the range 5 is greater or equal to finish 4')
	must_fail("testo run semantic/for.testo --test_spec wrong_start_finish_1", 	'Error: start of the range 0 is greater or equal to finish 0')

	must_fail("testo run semantic/misc.testo --test_spec empty_wait_text", 2)

def test_semantic_macro_tests():
	must_fail("testo run semantic/0011.testo", f"""{cwd}/semantic/0011.testo:7:1: In a macro call A()
{cwd}/semantic/0011.testo:2:2: Error: nested macro declarations are not supported""")

	must_fail("testo run semantic/0012.testo", f"""{cwd}/semantic/0012.testo:5:1: In a macro call A()
{cwd}/semantic/0012.testo:2:2: Error: param declaration inside macros is not supported""")

	must_fail("testo run semantic/0013.testo", f"""{cwd}/semantic/0013.testo:16:1: In a macro call B("VM1")
{cwd}/semantic/0013.testo:9:2: Error: test "VM1" is already defined

{cwd}/semantic/0013.testo:15:1: In a macro call A("VM1")
{cwd}/semantic/0013.testo:3:2: note: previous declaration was here""")

	must_fail("testo run semantic/0014.testo", f"""{cwd}/semantic/0014.testo:15:1: In a macro call B("VM1")
{cwd}/semantic/0014.testo:9:2: Error: virtual machine "VM1" is already defined

{cwd}/semantic/0014.testo:14:1: In a macro call A("VM1")
{cwd}/semantic/0014.testo:3:2: note: previous declaration was here""")

	must_fail("testo run semantic/0015.testo", f"""{cwd}/semantic/0015.testo:15:1: In a macro call B("VM1")
{cwd}/semantic/0015.testo:9:2: Error: flash drive "VM1" is already defined

{cwd}/semantic/0015.testo:14:1: In a macro call A("VM1")
{cwd}/semantic/0015.testo:3:2: note: previous declaration was here""")

	must_fail("testo run semantic/0016.testo", f"""{cwd}/semantic/0016.testo:15:1: In a macro call B("VM1")
{cwd}/semantic/0016.testo:9:2: Error: network "VM1" is already defined

{cwd}/semantic/0016.testo:14:1: In a macro call A("VM1")
{cwd}/semantic/0016.testo:3:2: note: previous declaration was here""")


	must_fail("testo run semantic/0017.testo", 'Error: unknown macro: A')
	must_fail("testo run semantic/0018.testo", 'Error: expected at least 2 args, 1 provided')
	must_fail("testo run semantic/0019.testo", 'Error: expected at most 4 args, 5 provided')
	must_fail("testo run semantic/0020.testo", 'Error: duplicate macro arg: arg1')
	must_fail("testo run semantic/0021.testo", 'Error: default value must be specified for macro arg arg3')
	must_fail("testo run semantic/0022.testo", 'Error: unexpected token IDENTIFIER, expected: }')
	must_fail("testo run semantic/0023.testo", 'Error: unexpected token IDENTIFIER, expected: }')
	must_fail("testo run semantic/0024.testo", 'Error: the "test_macro" macro does not contain commands, as expected')
