
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
