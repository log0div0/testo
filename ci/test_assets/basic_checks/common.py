
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

def must_succeed(cmd, out = None):
	p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
	stdout, stderr = p.communicate()
	stdout, stderr = stdout.decode('utf-8'), stderr.decode('utf-8')
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", cmd, " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
	print("STDOUT:", stdout)
	print("STDERR:", stderr)
	assert p.returncode == 0, "returncode == 0"
	if out is not None:
		assert out in stdout, f"STDOUT: {out}"
	return stdout, stderr

def must_fail(cmd, err = None, out = None):
	p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
	stdout, stderr = p.communicate()
	stdout, stderr = stdout.decode('utf-8'), stderr.decode('utf-8')
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", cmd, " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
	print("STDOUT:", stdout)
	print("STDERR:", stderr)
	if err is None:
		assert p.returncode != 0, "returncode != 0"
	elif isinstance(err, str):
		assert p.returncode != 0, "returncode != 0"
		assert err in stderr, f"STDERR: {err}"
	else:
		assert p.returncode == err, f"returncode == {err}"
	if out is not None:
		assert out in stdout, f"STDOUT: {out}"
	return stdout, stderr
