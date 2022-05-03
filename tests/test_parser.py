from common import *

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
