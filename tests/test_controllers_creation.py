from common import *

def test_vm_creation():
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_no_cpus", 						'Field "cpu" is not specified')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_no_ram", 						'Field "ram" is not specified')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_no_disk", 						'You must specify at least 1 disk')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_too_many_disks", 				'Too many IDE disks specified, maximum amount of IDE disks: 3')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_disk_no_size", 					'Either field "size" or "source" must be specified for the disk "main"')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_nic_no_attached_to", 			'Neither "attached_to" nor "attached_to_dev" is specified for the nic "my_nic"')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_nic_both_attached_to", 			'''Can't specify both "attached_to" and "attached_to_dev" for the same nic "my_nic"''')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_nic_incorrect_mac", 			'Incorrect mac address: "AA:BB:CC:DD:EE:FF:AA"')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_nic_incorrect_type", 			'NIC "my_nic" has unsupported adapter type: "some_type"')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_iso_does_not_exist", 			'Target iso file "/opt/some_iso.iso" does not exist')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_iso_does_not_exist_1", 			f'Target iso file "{cwd}/controllers_creation/../some_iso.iso" does not exist')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_loader_does_not_exist", 		'Target loader file "/opt/some_loader.fd" does not exist')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_image_does_not_exist", 			'Source disk image "/opt/some_image.img" does not exist')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_image_does_not_exist_1", 		f'Source disk image "{cwd}/controllers_creation/../some_image.img" does not exist')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_unknown_network", 				'NIC "my_nic" is attached to an unknown network: "some_network"')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec vm_attr_type_cant_convert", 				'Error: expected SIZE, but got NUMBER "4"')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_negative_attribute", 			'CPUs number must be a positive interger')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_negative_attribute_2", 			'CPUs number must be a positive interger')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_unknown_video_adapter_type", 	'Video "main_video" has unsupported adapter type: "some_video_adapter_type"')
	must_fail("testo run controllers_creation/vm_create.testo --test_spec create_vm_multiple_videos", 				'Multiple video devices are not supported at the moment')

def test_flash_creation():
	must_fail("testo run controllers_creation/flash_create.testo --test_spec flash_unexisting_folder", 				"Target folder /some/unexisting_folder doesn't exist")
	must_fail("testo run controllers_creation/flash_create.testo --test_spec flash_not_a_folder", 					"Specified folder /opt/ubuntu-16.04.6-server-amd64.iso is not a folder")
