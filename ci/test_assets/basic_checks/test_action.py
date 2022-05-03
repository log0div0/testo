from common import *

import os
import os.path

def test_action_qa():
	out, err = must_succeed("testo run action/action_qa.testo --assume_yes --test_spec exec_set_variable_2 --invalidate exec_set_variable_2")
	assert "Success 1" in out
	assert "Fail 1" not in out
	assert "Success 2" not in out
	assert "Fail 2" in out

	must_succeed("testo run action/action_qa.testo --assume_yes --test_spec exec_set_variable_1 --invalidate exec_set_variable_1", "hello from exec_set_variable_1!")

	out, err = must_fail("testo run action/action_qa.testo --assume_yes --test_spec exec_timeout")
	assert "Error while performing action exec bash" in out
	assert "Timeout was triggered" in out

	must_fail("testo run action/action_qa.testo --assume_yes --test_spec copyto_from_unexisting_runtime", out="Specified path doesn't exist: /some/unexisting_shit")

	# first make sure that the file does not exist!
	if os.path.exists("/tmp/something"):
		os.remove("/tmp/something")

	must_succeed("testo run action/action_qa.testo --assume_yes --test_spec copyto_nocheck --invalidate copyto_nocheck")
