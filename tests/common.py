
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

def must_succeed(cmd, out=None, err=None, input=None):
	p = Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE, shell=True)
	if input is not None:
		input = input.encode('utf-8')
	stdout, stderr = p.communicate(input)
	stdout, stderr = stdout.decode('utf-8'), stderr.decode('utf-8')
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", cmd, " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
	print("STDOUT:", stdout)
	print("STDERR:", stderr)
	assert p.returncode == 0, "returncode == 0"
	if out is not None:
		assert out in stdout, f"STDOUT: {out}"
	if err is not None:
		assert err in stdout, f"STDERR: {err}"
	return stdout, stderr

def must_fail(cmd, err=None, out=None, input=None):
	p = Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE, shell=True)
	if input is not None:
		input = input.encode('utf-8')
	stdout, stderr = p.communicate(input)
	stdout, stderr = stdout.decode('utf-8'), stderr.decode('utf-8')
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", cmd, " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
	print("STDOUT:", stdout)
	print("STDERR:", stderr)
	assert p.returncode != 0, "returncode != 0"
	if out is not None:
		assert out in stdout, f"STDOUT: {out}"
	if err is not None:
		if isinstance(err, str):
			assert err in stderr, f"STDERR: {err}"
		else:
			assert p.returncode == err, f"returncode == {err}"
	return stdout, stderr
