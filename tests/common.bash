
setup() {
	rm -rf ./dummy_hypervisor_files
	rm -rf ./flash_drives_metadata
	rm -rf ./vm_metadata

	mkdir ./dummy_hypervisor_files

	TESTO_BIN=$SBIN_DIR/testo
}
