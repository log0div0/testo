from common import *

import os
import os.path

def test_action_qa():
	out, err = must_fail("testo run action/action_qa.testo --assume_yes --test_spec exec_timeout")
	assert "Error while performing action exec bash" in out
	assert "Timeout was triggered" in out

	must_succeed("testo run action/action_qa.testo --assume_yes --test_spec exec_set_variable --invalidate exec_set_variable", "ура, товарищи!")
	must_fail("testo run action/action_qa.testo --assume_yes --test_spec copyto_from_unexisting_runtime", out="Specified path doesn't exist: /some/unexisting_shit")

	# first make sure that the file does not exist!
	if os.path.exists("/tmp/something"):
		os.remove("/tmp/something")

	must_succeed("testo run action/action_qa.testo --assume_yes --test_spec copyto_nocheck --invalidate copyto_nocheck")