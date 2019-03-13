
import negotiator, logging, threading

def main():
	agent = negotiator.GuestChannel("/dev/virtio-ports/negotiator.0")
	agent.main_loop()

if __name__ == "__main__":
	try:

		logging.basicConfig(filename='/var/log/negotiator.log', level=logging.DEBUG)

		main()
	except Exception as error:
		logging.error(error)
