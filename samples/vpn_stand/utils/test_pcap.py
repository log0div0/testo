
from scapy.all import *

packets = rdpcap('/tmp/captured.pcapng')

for pkt in packets:
	if ICMP in pkt:
		exit(1)