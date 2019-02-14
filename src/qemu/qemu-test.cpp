
#include <iostream>
#include <libvirt/libvirt.h>
#include <libvirt/virterror.h>

int main() {
	std::cout << "Hello world\n";
	auto conn = virConnectOpenAuth("qemu:///system", virConnectAuthPtrDefault, 0);
	if (!conn) {
		std::cout << "Error: " << virGetLastErrorMessage() << std::endl;
		return 0;
	}
	std::cout << "Connected" << std::endl;

	virDomainPtr *nameList = NULL;

	int flags = VIR_CONNECT_LIST_DOMAINS_ACTIVE |
                VIR_CONNECT_LIST_DOMAINS_INACTIVE;

	auto numNames = virConnectListAllDomains(conn, &nameList, flags);

	for (auto i = 0; i < numNames; i++) {
		std::cout << "NAME: " << virDomainGetName(nameList[i]) << std::endl;
		virDomainFree(nameList[i]);
	}

	delete nameList;

	virConnectClose(conn);


}