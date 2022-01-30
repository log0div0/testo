
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
