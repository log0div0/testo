
import json
import time
import base64
import subprocess
import os
import socket
import logging
import struct

class ProtocolError(Exception):
	pass

class RemoteMethodFailed(Exception):
	pass

# -------------------------------------------------
# Channel
# -------------------------------------------------
class Channel(object):

	def __init__(self, handle):
		self.handle = handle

	def read(self):

		while True:
			bytes_count_str = self.handle.read(4)
			if len(bytes_count_str) == 0:
				time.sleep(0.1)
			else:
				break
		bytes_count_array = bytearray()
		bytes_count_array.extend(bytes_count_str)
		json_size = struct.unpack('<I',bytes_count_array)[0]
		encoded_value = self.handle.read(json_size)
		try:
			decoded_value = json.loads(encoded_value)
			return decoded_value
		except Exception as e:
			raise ProtocolError("Expecting a json, got '%s'" % encoded_value)

	def write(self, decoded_value):
		encoded_message = json.dumps(decoded_value)
		bytes_count = len(encoded_message)
		bytes_count_array = struct.pack('<I', bytes_count)
		self.handle.write("%s%s" % (bytes_count_array, encoded_message))
		self.handle.flush()

# -------------------------------------------------
# GuestChannel
# -------------------------------------------------
class GuestChannel(Channel):
	def __init__(self, device):

		logging.debug(u'Negotiator is running on port %s' % (device))
		Channel.__init__(self, open(device, 'r+'))

	def main_loop(self):

		while True:
			request = self.read()
			method_name = request.get('method')
			method = getattr(self, method_name, None)
			logging.debug(u'%s method called:' % (method_name))
			args = request.get('args', [])
			kw = request.get('kw', {})
			if method:
				try:
					result = method(*args, **kw)
					self.write(dict(success=True, result=result))
				except Exception as e:
					self.write(dict(success=False, error=str(e)))
			else:
				self.write(dict(success=False, error="Method %s not supported" % method_name))

	def check_avaliable(self, *args, **kwargs):
		pass

	def copy_file(self, file, **kwargs):
		subprocess.check_call(["mkdir", "-p", os.path.dirname(file["path"])])

		logging.info(u'Copy file %s ...' % file["path"])
		with open(file["path"], "wb") as f:
			f.write(base64.b64decode(file["content"]))

	def copy_files_out(self, src, dst):
		if not os.path.exists(src):
			raise Exception("target file/folder does not exist!")

		if os.path.isfile(src):
			with open(src, "rb") as f:
				data = f.read()

			file = {
				"path": dst,
				"content": base64.b64encode(data)
			}
			return [file]
		elif os.path.isdir(src):
			files = []
			for file in os.listdir(src):
				if os.path.isdir(src + "/" + file):
					#no content indicates that this is a directory
					files.append({
						"path": dst + "/" + file
					})
				files += self.copy_files_out(src + "/" + file, dst + "/" + file)
			return files
		else:
			return []

	def execute(self, command, timeout, **kwargs):
		logging.info(u'Execute command "%s" ...\n' % command)
		p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

		while True:
			line = p.stdout.readline()
			if not line:
				break
			result = {}
			result["stdout"] = line
			result["status"] = "pending"
			self.write(dict(success=True, result = result))

		p.wait()

		result = {}
		result["status"] = "finished"
		result["exit_code"] = p.returncode
		return result
