# Часть 7. Связываем две машины по сети

## С чем Вы познакомитесь

В этой части вы:

1. Познакомитесь с управлением несколькими виртуальными машинами в одном тесте
2. Познакомитесь с действием `plug/unplug nic`
3. Узнаете еще несколько атрибутов сетевых адаптеров в виртуальных машинах
4. Завершите ознакомление с виртуальными сетями

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. На хостовой машине имеется прямой (без прокси) доступ в Интернет
4. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствуующим образом поправить тестовый скрипт.
5. Имеется образ с гостевыми дополнениями Testo в одной папке с установочным образом Ubuntu.
6. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
7. (Рекомендовано) Проделаны шаги из [шестой части](6_nat).

## Вступление

В прошлой части мы познакомились с одним из возможных применений виртуальных сетей - доступ в Интернет. Но, конечно же, виртуальные сети также можно (и нужно) применять для соединения нескольких виртуальных машин между собой. Именно этим мы и займёмся в этом уроке, и попутно познакомимся с несколькими небольшими уловками чтобы сделать тесты более удобными и читаемыми.

В конце этого урока мы сможем воспроизвести такой стенд.

![Primary NIC](/static/tutorials/7_ping/network.png)

## С чего начать

С этого момента в наших тестовых сценариях появятся две виртуальных машины, которые будут условно играть роль клиента и сервера. Для того, чтобы наши тестовые сценарии остались читаемыми, нам нужно провести небольшой рефакторинг:

1. Переименовать машину `my_ubuntu` в `server`;
2. Переименовать параметры `hostname`, `login`, `password` в `server_hostname`, `server_login` и `default_password` соответственно и подправить все места где к ним происходит обращение;
3. Переименовать тесты `ubuntu_installation` и `guest_additions_installation` в `server_install_ubuntu` и `server_install_guest_additions` соответственно;
4. Удалить тест `check_internet`.

В итоге должен получиться такой скрипт:

	network internet {
		mode: "nat"
	}

	machine server {
		cpus: 1
		ram: 512Mb
		disk_size: 5Gb
		iso: "${ISO_DIR}/ubuntu_server.iso"

		nic nat: {
			attached_to: "internet"
		}
	}

	param server_hostname "server"
	param server_login "server-login"
	param default_password "1111"

	test server_install_ubuntu {
		server {
			start
			wait "English"
			press Enter
			#Действия могут разделяться символом новой строки
			#или точкой с запятой
			wait "Install Ubuntu Server"; press Enter;
			wait "Choose the language";	press Enter
			wait "Select your location"; press Enter
			wait "Detect keyboard layout?";	press Enter
			wait "Country of origin for the keyboard"; press Enter
			wait "Keyboard layout"; press Enter
			#wait "No network interfaces detected" timeout 5m; press Enter
			wait "Hostname:" timeout 5m; press Backspace*36; type "${server_hostname}"; press Enter
			wait "Full name for the new user"; type "${server_login}"; press Enter
			wait "Username for your account"; press Enter
			wait "Choose a password for the new user"; type "${default_password}"; press Enter
			wait "Re-enter password to verify"; type "${default_password}"; press Enter
			wait "Use weak password?"; press Left, Enter
			wait "Encrypt your home directory?"; press Enter
			
			#wait "Select your timezone" timeout 2m; press Enter
			wait "Is this time zone correct?" timeout 2m; press Enter
			wait "Partitioning method"; press Enter
			wait "Select disk to partition"; press Enter
			wait "Write the changes to disks and configure LVM?"; press Left, Enter
			wait "Amount of volume group to use for guided partitioning"; press Enter
			wait "Write the changes to disks?"; press Left, Enter
			wait "HTTP proxy information" timeout 3m; press Enter
			wait "How do you want to manage upgrades" timeout 6m; press Enter
			wait "Choose software to install"; press Enter
			wait "Install the GRUB boot loader to the master boot record?" timeout 10m; press Enter
			wait "Installation complete" timeout 1m; 

			unplug dvd; press Enter
			wait "server_login:" timeout 2m; type "${server_login}"; press Enter
			wait "Password:"; type "${default_password}"; press Enter
			wait "Welcome to Ubuntu"
		}
	}

	param guest_additions_pkg "testo-guest-additions*"
	test server_install_guest_additions: server_install_ubuntu {
		server {
			plug dvd "${ISO_DIR}/testo-guest-additions.iso"

			type "sudo su"; press Enter;
			#Обратите внимание, обращаться к параметрам можно в любом участке строки
			wait "password for ${server_login}"; type "${default_password}"; press Enter
			wait "root@${server_hostname}"

			type "mount /dev/cdrom /media"; press Enter
			wait "mounting read-only"; type "dpkg -i /media/${guest_additions_pkg}"; press Enter;
			wait "Setting up testo-guest-additions"
			type "umount /media"; press Enter;
			#Дадим немного времени для команды umount
			sleep 2s
			unplug dvd
		}
	}

Т.к. мы переименовали виртуальную машину в `server`, то для Testo эта машина выглядит как абсолютно новая сущность, и в итоге весь тестовый процесс будет пройден заново, в том числе и создание виртуальной машины.

При этом старая виртуальная машина, `my_ubuntu`, **никуда сама по себе не денется**. Платформа Testo удет считать, что эта машина еще может вам когда-нибудь пригодиться и удалять её не нужно. Но т.к. мы понимаем, что нам эта машина больше не нужна, давайте мы ее сами удалим.

<Terminal height="150px">
	<span className="">user$ sudo testo clean<br/></span>
	<span className="">Deleted network internet<br/></span>
	<span className="">Deleted virtual machine my_ubuntu<br/></span>
	<span className="">user$ </span>
</Terminal>

Можно заметить, что помимо виртуальлной машины была также удалена виртуальная сеть `internet`, которой мы пользовались.

Теперь, прежде чем продолжать что-то изменять, хорошо было бы убедиться, что мы своими изменениями ничего не сломали.

<Terminal height="600px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">server_install_ubuntu<br/></span>
	<span className="blue ">[  0%] Restoring snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">server_install_ubuntu<br/></span>
	<span className="blue ">[  0%] Starting virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">English </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Install Ubuntu Server </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose the language </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Select your location </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Detect keyboard layout? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Country of origin for the keyboard </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Keyboard layout </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Hostname: </span>
	<span className="blue ">for 5m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">BACKSPACE </span>
	<span className="blue ">36 times </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"server" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Full name for the new user </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"server-login" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Username for your account </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose a password for the new user </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Re-enter password to verify </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Use weak password? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">LEFT </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Encrypt your home directory? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Is this time zone correct? </span>
	<span className="blue ">for 2m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Partitioning method </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Select disk to partition </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Write the changes to disks and configure LVM? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">LEFT </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Amount of volume group to use for guided partitioning </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Write the changes to disks? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">LEFT </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">HTTP proxy information </span>
	<span className="blue ">for 3m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">How do you want to manage upgrades </span>
	<span className="blue ">for 6m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose software to install </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Install the GRUB boot loader to the master boot record? </span>
	<span className="blue ">for 10m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Installation complete </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Unplugging dvd </span>
	<span className="yellow "> </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">server login: </span>
	<span className="blue ">for 2m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"server-login" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Password: </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Welcome to Ubuntu </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Taking snapshot </span>
	<span className="yellow ">server_install_ubuntu</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[ 50%] Test </span>
	<span className="yellow bold">server_install_ubuntu</span>
	<span className="green bold"> PASSED in 0h:5m:35s<br/></span>
	<span className="blue ">[ 50%] Preparing the environment for test </span>
	<span className="yellow ">server_install_guest_additions<br/></span>
	<span className="blue ">[ 50%] Running test </span>
	<span className="yellow ">server_install_guest_additions<br/></span>
	<span className="blue ">[ 50%] Plugging dvd </span>
	<span className="yellow ">/opt/iso/testo-guest-additions.iso </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"sudo su" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Waiting </span>
	<span className="yellow ">password for server-login </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Waiting </span>
	<span className="yellow ">root@server </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"mount /dev/cdrom /media" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Waiting </span>
	<span className="yellow ">mounting read-only </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"dpkg -i /media/testo-guest-additions*" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Waiting </span>
	<span className="yellow ">Setting up testo-guest-additions </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Typing </span>
	<span className="yellow ">"umount /media" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 2s<br/></span>
	<span className="blue ">[ 50%] Unplugging dvd </span>
	<span className="yellow "> </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 50%] Taking snapshot </span>
	<span className="yellow ">server_install_guest_additions</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">server_install_guest_additions</span>
	<span className="green bold"> PASSED in 0h:0m:15s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 2 TESTS IN 0h:5m:51s<br/></span>
	<span className="blue bold">UP-TO-DATE: 0<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 2<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

## Создаём вторую машину

Пришло время создать вторую виртуальную машину. Как вы уже догадались, называться она будет `client` и будет, по большей части, копией машины `server`

	machine client {
		cpus: 1
		ram: 512Mb
		disk_size: 5Gb
		iso: "${ISO_DIR}/ubuntu_server.iso"

		nic nat: {
			attached_to: "internet"
		}
	}

Но если мы оставим виртуальные машины в таком виде, они не будут связаны между собой. Для того, чтобы добавить между ними связность, нам необходимо объявить новую виртуальную сеть

	network LAN {
		mode: "internal"
	}

Обратите внимание, что эта сеть работает уже в режиме `internal`, то есть предназначена для внутреннего взаимодействия между машинами, без доступа во внешнюю среду.

Осталось нам лишь добавить в виртуальные машины сетевые адаптеры, которые будут подлючаться к новой сети `LAN`:

	machine client {
		cpus: 1
		ram: 512Mb
		disk_size: 5Gb
		iso: "${ISO_DIR}/ubuntu_server.iso"

		nic nat: {
			attached_to: "internet"
		}

		nic server_side: {
			attached_to: "LAN"
			mac: "52:54:00:00:00:AA"
		}
	}


	machine server {
		cpus: 1
		ram: 512Mb
		disk_size: 5Gb
		iso: "${ISO_DIR}/ubuntu_server.iso"

		nic nat: {
			attached_to: "internet"
		}

		nic client_side: {
			attached_to: "LAN"
			mac: "52:54:00:00:00:BB"
		}
	}

Обратите внимание, что для "внутренних" сетевых адаптеров мы указали еще атрибут `mac`. Указание точного MAC-адреса позволит нам чуть позже переименовать сетевые интерфейсы так, чтобы в них можно было легко ориентироваться внутри тестовых сценариев.

Что ж, добавим два новых теста уже для машины `client`: `client_install_ubuntu` и `client_install_guest_additions`. Не забудем также добавить несколько новых параметров.

	param client_hostname "client"
	param client_login "client-login"

	test client_install_ubuntu {
		client {
			start
			wait "English"
			press Enter
			#Действия могут разделяться символом новой строки
			#или точкой с запятой
			wait "Install Ubuntu Server"; press Enter;
			wait "Choose the language";	press Enter
			wait "Select your location"; press Enter
			wait "Detect keyboard layout?";	press Enter
			wait "Country of origin for the keyboard"; press Enter
			wait "Keyboard layout"; press Enter
			#wait "No network interfaces detected" timeout 5m; press Enter
			wait "Hostname:" timeout 5m; press Backspace*36; type "${client_hostname}"; press Enter
			wait "Full name for the new user"; type "${client_login}"; press Enter
			wait "Username for your account"; press Enter
			wait "Choose a password for the new user"; type "${default_password}"; press Enter
			wait "Re-enter password to verify"; type "${default_password}"; press Enter
			wait "Use weak password?"; press Left, Enter
			wait "Encrypt your home directory?"; press Enter
			
			#wait "Select your timezone" timeout 2m; press Enter
			wait "Is this time zone correct?" timeout 2m; press Enter
			wait "Partitioning method"; press Enter
			wait "Select disk to partition"; press Enter
			wait "Write the changes to disks and configure LVM?"; press Left, Enter
			wait "Amount of volume group to use for guided partitioning"; press Enter
			wait "Write the changes to disks?"; press Left, Enter
			wait "HTTP proxy information" timeout 3m; press Enter
			wait "How do you want to manage upgrades" timeout 6m; press Enter
			wait "Choose software to install"; press Enter
			wait "Install the GRUB boot loader to the master boot record?" timeout 10m; press Enter
			wait "Installation complete" timeout 1m; 

			unplug dvd; press Enter
			wait "${client_hostname} login:" timeout 2m; type "${client_login}"; press Enter
			wait "Password:"; type "${default_password}"; press Enter
			wait "Welcome to Ubuntu"
		}
	}

	test client_install_guest_additions: client_install_ubuntu {
		client {
			plug dvd "${ISO_DIR}/testo-guest-additions.iso"

			type "sudo su"; press Enter;
			#Обратите внимание, обращаться к параметрам можно в любом участке строки
			wait "password for ${client_login}"; type "${default_password}"; press Enter
			wait "root@${client_hostname}"

			type "mount /dev/cdrom /media"; press Enter
			wait "mounting read-only"; type "dpkg -i /media/${guest_additions_pkg}"; press Enter;
			wait "Setting up testo-guest-additions"
			type "umount /media"; press Enter;
			#Дадим немного времени для команды umount
			sleep 2s
			unplug dvd
		}
	}

Не стоит пока пугаться большого количества дублирующегося кода. В будущем мы познакомимся со способом объединять одинаковые действие в именованные блоки (макросы) и эти сценарии станут намного компакнтнее

Теперь попробуем запустить все наши тестовые сценарии.

<Terminal height="700px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="">Some tests have lost their cache:<br/></span>
	<span className="">	- server_install_ubuntu<br/></span>
	<span className="">Do you confirm running them and all their children? [y/N]: y<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="blue ">[  0%] Preparing the environment for test </span>
	<span className="yellow ">server_install_ubuntu<br/></span>
	<span className="blue ">[  0%] Creating virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Taking snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Running test </span>
	<span className="yellow ">server_install_ubuntu<br/></span>
	<span className="blue ">[  0%] Starting virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">English </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Install Ubuntu Server </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Choose the language </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Select your location </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Detect keyboard layout? </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Country of origin for the keyboard </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Keyboard layout </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[  0%] Waiting </span>
	<span className="yellow ">Hostname: </span>
	<span className="blue ">for 5m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="red bold">/home/alex/testo/hello_world.testo:61:3: Error while performing action wait Hostname: timeout 5m on virtual machine server:<br/>	-Timeout<br/></span>
	<span className="red bold">[ 25%] Test </span>
	<span className="yellow bold">server_install_ubuntu</span>
	<span className="red bold"> FAILED in 0h:5m:12s<br/></span>
	<span className="">user$ </span>
</Terminal>

Но у нас сломалась установка Ubuntu. Почему? Потому что мы добавили второй сетевой адаптер в виртуальную машину и теперь у нас появился новый экран, которого мы не ожидали ранее.

![Primary NIC](/static/tutorials/7_ping/primary_nic.png)

В качестве главного интерфейса нам нужно выбрать первый (то есть нажать Enter). Давайте подправим наш установочный сценарий.

	wait "Keyboard layout"; press Enter
	#wait "No network interfaces detected" timeout 5m; press Enter
	wait "Primary network interface"; press Enter
	wait "Hostname:" timeout 5m; press Backspace*36; type "${client_hostname}"; press Enter

После этого можно еще раз запустить тесты и они должны отработать успешно.

## Вытаскиваем ненужный адаптер

Теперь у нас имеются две виртуальные машины, которые соединены виртуальной сетью `LAN`. Но прежде чем настраивать IP-адреса и проверять связность наших машин, было бы неплохо для начала разобраться с сетевым адаптером `nat`.

В начале тестовых сценариев, когда мы только подготавливаем виртуальные машины к работе, соединение в Интернетом, безусловно, очень важно: оно позволяет установить какие-то недостающие пакеты на систему. Но в дальнейшем это соединение с Интернетом будет лишь мешать, поэтому от него хотелось бы избавиться.

В платформе Testo есть две возможности по управлению сетевыми адаптерами:

1. Включать/выключать "сетевой провод" из сетевого адаптера ([`plug/unplug link`](/docs/lang/actions#plug-link))
2. Включать/выключать сетевой адаптер целиком ([`plug/unplug nic`](/docs/lang/actions#plug-nic))

Включать и выключать "сетевой провод" можно во время работы виртуальной машины, а вот для включения/отключения сетевого адаптера необходимо сначала выключить виртуальную машину.

Давайте посмотрим на примере машины `server`. Создадим новый тест `server_unplug_nat`.

	test server_unplug_nat: server_install_guest_additions {
		server {
			shutdown
			unplug nic nat
			start

			wait "${server_hostname} login:" timeout 2m; type "${server_login}"; press Enter
			wait "Password:"; type "${default_password}"; press Enter
			wait "Welcome to Ubuntu"
		}
	}

Этот тест мы начинаем с того, что останавливаем виртуальную машину. Вообще, есть два способа остановить виртуальную машину извне: действие [`stop`](/docs/lang/actions#stop) (аналог "выдергивания шнура питания" из виртуальной машины) и действие [`shutdown`](/docs/lang/actions#shutdown) (аналог нажатия на кнопку питания на системном блоке). Действие `shutdown` несколько предпочтительнее.

После Выключения питания мы вытаскиваем (прям на уровне PCI-устройства) сетевой адаптер `nat` и запускаем машину обратно. Заканчивается тест после успешного логина.

<Terminal height="650px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --test_spec server_unplug_nat<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">server_unplug_nat<br/></span>
	<span className="blue ">[ 67%] Restoring snapshot </span>
	<span className="yellow ">server_install_guest_additions</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">server_unplug_nat<br/></span>
	<span className="blue ">[ 67%] Shutting down virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 1m<br/></span>
	<span className="blue ">[ 67%] Unplugging nic </span>
	<span className="yellow ">nat </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Starting virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Waiting </span>
	<span className="yellow ">server login: </span>
	<span className="blue ">for 2m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Typing </span>
	<span className="yellow ">"server-login" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Waiting </span>
	<span className="yellow ">Password: </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Waiting </span>
	<span className="yellow ">Welcome to Ubuntu </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 67%] Taking snapshot </span>
	<span className="yellow ">server_unplug_nat</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">server_unplug_nat</span>
	<span className="green bold"> PASSED in 0h:0m:28s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:28s<br/></span>
	<span className="blue bold">UP-TO-DATE: 2<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

После выполнения теста можно зайти в свойства виртуальной машины и убедиться, что в неё остался лишь один сетевой адаптер.

Также давайте продублируем этот тест для машины `client`

	test client_unplug_nat: client_install_guest_additions {
		client {
			shutdown
			unplug nic nat
			start

			wait "${client_hostname} login:" timeout 2m; type "${client_login}"; press Enter
			wait "Password:"; type "${default_password}"; press Enter
			wait "Welcome to Ubuntu"
		}
	}

<Terminal height="650px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --test_spec client_unplug_nat<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="blue ">[ 67%] Preparing the environment for test </span>
	<span className="yellow ">client_unplug_nat<br/></span>
	<span className="blue ">[ 67%] Restoring snapshot </span>
	<span className="yellow ">client_install_guest_additions</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Running test </span>
	<span className="yellow ">client_unplug_nat<br/></span>
	<span className="blue ">[ 67%] Shutting down virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 1m<br/></span>
	<span className="blue ">[ 67%] Unplugging nic </span>
	<span className="yellow ">nat </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Starting virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Waiting </span>
	<span className="yellow ">client login: </span>
	<span className="blue ">for 2m with interval 1s in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Typing </span>
	<span className="yellow ">"client-login" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Waiting </span>
	<span className="yellow ">Password: </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Typing </span>
	<span className="yellow ">"1111" </span>
	<span className="blue ">with interval 30ms in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Pressing key </span>
	<span className="yellow ">ENTER </span>
	<span className="blue ">on virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Waiting </span>
	<span className="yellow ">Welcome to Ubuntu </span>
	<span className="blue ">for 1m with interval 1s in virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 67%] Taking snapshot </span>
	<span className="yellow ">client_unplug_nat</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">client_unplug_nat</span>
	<span className="green bold"> PASSED in 0h:0m:28s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 3 TESTS IN 0h:0m:28s<br/></span>
	<span className="blue bold">UP-TO-DATE: 2<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Теперь у нас есть виртуальные машины без лишних сетевых интерфейсов и соединенные одной виртуальной сетью. Перед финальной настройкой остался лишь один подготовительный шаг.

## Переименовываем сетевые интерфейсы

По умолчанию сетевые адаптеры создаются в виртуальной машине с довольно неинформативными названиями (например, `ens3` и `ens4`) при том, что для нас в тестовых сценариях было бы гораздо удобнее, если бы интерфейсы назывались так же, как мы их назвали в объявлении виртуальной машины: `client_side` и `server_side`. Для этого можно переименовать сетевые адаптеры, используя в качестве основы MAC-адрес, который нам заранее известен.

Для этого можно использовать следующий баш-скрипт (ссылка на него будет доступна в конце урока).

``` bash
#!/bin/bash

set -e

mac=$1

oldname=`ifconfig -a | grep ${mac,,} | awk '{print $1}'`
newname=$2

echo SUBSYSTEM==\"net\", ACTION==\"add\", ATTR{address}==\"$mac\", NAME=\"$newname\", DRIVERS==\"?*\" >> /lib/udev/rules.d/70-test-tools.rules

rm -f /etc/network/interfaces
echo source /etc/network/interfaces.d/* >> /etc/network/interfaces
echo auto lo >> /etc/network/interfaces
echo iface lo inet loopback >> /etc/network/interfaces

ip link set $oldname down
ip link set $oldname name $newname
ip link set $newname up

echo "Renaming success"
```
Давайте сохраним этот скрипт в файл `rename_net.sh` и сохраним его в ту же папку, что и файл с тестовы сценарием `hello_world.testo`

После этого напишем новый тест для сервера `server_prepare`

	test server_prepare: server_unplug_nat {
		server {
			copyto "./rename_net.sh" "/opt/rename_net.sh"
			exec bash """
				chmod +x /opt/rename_net.sh
				/opt/rename_net.sh 52:54:00:00:00:bb client_side
				ip ad 
			"""
		}
	}

<Terminal height="600px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --test_spec server_prepare<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="blue ">[ 75%] Preparing the environment for test </span>
	<span className="yellow ">server_prepare<br/></span>
	<span className="blue ">[ 75%] Restoring snapshot </span>
	<span className="yellow ">server_unplug_nat</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 75%] Running test </span>
	<span className="yellow ">server_prepare<br/></span>
	<span className="blue ">[ 75%] Copying </span>
	<span className="yellow ">./rename_net.sh </span>
	<span className="blue ">to virtual machine </span>
	<span className="yellow ">server </span>
	<span className="blue ">to destination </span>
	<span className="yellow ">/opt/rename_net.sh </span>
	<span className="blue ">with timeout 10m<br/></span>
	<span className="blue ">[ 75%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ chmod +x /opt/rename_net.sh<br/></span>
	<span className=" ">+ /opt/rename_net.sh 52:54:00:00:00:bb client_side<br/></span>
	<span className=" ">Renaming success<br/></span>
	<span className=" ">+ ip ad<br/></span>
	<span className=" ">1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1<br/></span>
	<span className=" ">    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
	<span className=" ">    inet 127.0.0.1/8 scope host lo<br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">    inet6 ::1/128 scope host <br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">2: client_side: &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt; mtu 1500 qdisc pfifo_fast state UP group default qlen 1000<br/></span>
	<span className=" ">    link/ether 52:54:00:00:00:bb brd ff:ff:ff:ff:ff:ff<br/></span>
	<span className=" ">    inet6 fe80::5054:ff:fe00:bb/64 scope link tentative <br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className="blue ">[ 75%] Taking snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">server_prepare</span>
	<span className="green bold"> PASSED in 0h:0m:2s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 4 TESTS IN 0h:0m:2s<br/></span>
	<span className="blue bold">UP-TO-DATE: 3<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

В выводе команды `ip` видно, что теперь наш оставшийся сетевой адаптер называется вполне понятно и лаконично: `client_side`.

Осталось лишь добавить сетевые настройки в этот сетевой интерфейс.

	test server_prepare: server_unplug_nat {
		server {
			copyto "./rename_net.sh" "/opt/rename_net.sh"
			exec bash """
				chmod +x /opt/rename_net.sh
				/opt/rename_net.sh 52:54:00:00:00:bb client_side
				ip a a 192.168.1.1/24 dev client_side
				ip l s client_side up
				ip ad
			"""
		}
	}

И повторяем такие же действия для машины `client`

<Terminal height="700px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="">Some tests have lost their cache:<br/></span>
	<span className="">	- server_prepare<br/></span>
	<span className="">Do you confirm running them and all their children? [y/N]: y<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="blue ">[ 75%] Preparing the environment for test </span>
	<span className="yellow ">server_prepare<br/></span>
	<span className="blue ">[ 75%] Restoring snapshot </span>
	<span className="yellow ">server_unplug_nat</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 75%] Running test </span>
	<span className="yellow ">server_prepare<br/></span>
	<span className="blue ">[ 75%] Copying </span>
	<span className="yellow ">./rename_net.sh </span>
	<span className="blue ">to virtual machine </span>
	<span className="yellow ">server </span>
	<span className="blue ">to destination </span>
	<span className="yellow ">/opt/rename_net.sh </span>
	<span className="blue ">with timeout 10m<br/></span>
	<span className="blue ">[ 75%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ chmod +x /opt/rename_net.sh<br/></span>
	<span className=" ">+ /opt/rename_net.sh 52:54:00:00:00:bb client_side<br/></span>
	<span className=" ">Renaming success<br/></span>
	<span className=" ">+ ip a a 192.168.1.1/24 dev client_side<br/></span>
	<span className=" ">+ ip l s client_side up<br/></span>
	<span className=" ">+ ip ad<br/></span>
	<span className=" ">1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1<br/></span>
	<span className=" ">    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
	<span className=" ">    inet 127.0.0.1/8 scope host lo<br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">    inet6 ::1/128 scope host <br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">2: client_side: &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt; mtu 1500 qdisc pfifo_fast state UP group default qlen 1000<br/></span>
	<span className=" ">    link/ether 52:54:00:00:00:bb brd ff:ff:ff:ff:ff:ff<br/></span>
	<span className=" ">    inet 192.168.1.1/24 scope global client_side<br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">    inet6 fe80::5054:ff:fe00:bb/64 scope link tentative <br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className="blue ">[ 75%] Taking snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[ 88%] Test </span>
	<span className="yellow bold">server_prepare</span>
	<span className="green bold"> PASSED in 0h:0m:4s<br/></span>
	<span className="blue ">[ 88%] Preparing the environment for test </span>
	<span className="yellow ">client_prepare<br/></span>
	<span className="blue ">[ 88%] Restoring snapshot </span>
	<span className="yellow ">client_unplug_nat</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 88%] Running test </span>
	<span className="yellow ">client_prepare<br/></span>
	<span className="blue ">[ 88%] Copying </span>
	<span className="yellow ">./rename_net.sh </span>
	<span className="blue ">to virtual machine </span>
	<span className="yellow ">client </span>
	<span className="blue ">to destination </span>
	<span className="yellow ">/opt/rename_net.sh </span>
	<span className="blue ">with timeout 10m<br/></span>
	<span className="blue ">[ 88%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ chmod +x /opt/rename_net.sh<br/></span>
	<span className=" ">+ /opt/rename_net.sh 52:54:00:00:00:aa server_side<br/></span>
	<span className=" ">Renaming success<br/></span>
	<span className=" ">+ ip a a 192.168.1.2/24 dev server_side<br/></span>
	<span className=" ">+ ip l s server_side up<br/></span>
	<span className=" ">+ ip ad<br/></span>
	<span className=" ">1: lo: &lt;LOOPBACK,UP,LOWER_UP&gt; mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1<br/></span>
	<span className=" ">    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00<br/></span>
	<span className=" ">    inet 127.0.0.1/8 scope host lo<br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">    inet6 ::1/128 scope host <br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">2: server_side: &lt;BROADCAST,MULTICAST,UP,LOWER_UP&gt; mtu 1500 qdisc pfifo_fast state UP group default qlen 1000<br/></span>
	<span className=" ">    link/ether 52:54:00:00:00:aa brd ff:ff:ff:ff:ff:ff<br/></span>
	<span className=" ">    inet 192.168.1.2/24 scope global server_side<br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className=" ">    inet6 fe80::5054:ff:fe00:aa/64 scope link tentative <br/></span>
	<span className=" ">       valid_lft forever preferred_lft forever<br/></span>
	<span className="blue ">[ 88%] Taking snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">client_prepare</span>
	<span className="green bold"> PASSED in 0h:0m:4s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 8 TESTS IN 0h:0m:8s<br/></span>
	<span className="blue bold">UP-TO-DATE: 6<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 2<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

## Пингуем!

Наконец, всё готово и мы можем проверить доступность двух машин друг для друга.

Для этого создаём еще один тест `test_ping`, который, в отличие от всех предыдущих наших тестов, будет отнаследован сразу **от двух** родительских тестов, т.к. нам необходимо чтобы были выполнены тесты и `client_prepare` и `server_prepare`

	test test_ping: client_prepare, server_prepare {
		client exec bash "ping 192.168.1.2 -c5"
		server exec bash "ping 192.168.1.1 -c5"
	}

Попробуем запустить этот тест.


<Terminal height="600px">
	<span className="">user$ sudo testo run hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --test_spec test_ping<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">test_ping<br/></span>
	<span className="blue ">[ 89%] Preparing the environment for test </span>
	<span className="yellow ">test_ping<br/></span>
	<span className="blue ">[ 89%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 89%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 89%] Running test </span>
	<span className="yellow ">test_ping<br/></span>
	<span className="blue ">[ 89%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ ping 192.168.1.2 -c5<br/></span>
	<span className=" ">PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.056 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.036 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.046 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.046 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.043 ms<br/></span>
	<span className=" "><br/></span>
	<span className=" ">--- 192.168.1.2 ping statistics ---<br/></span>
	<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
	<span className=" ">rtt min/avg/max/mdev = 0.036/0.045/0.056/0.008 ms<br/></span>
	<span className="blue ">[ 89%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ ping 192.168.1.1 -c5<br/></span>
	<span className=" ">PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.057 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.038 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.042 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.043 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.042 ms<br/></span>
	<span className=" "><br/></span>
	<span className=" ">--- 192.168.1.1 ping statistics ---<br/></span>
	<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
	<span className=" ">rtt min/avg/max/mdev = 0.038/0.044/0.057/0.008 ms<br/></span>
	<span className="blue ">[ 89%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 89%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">test_ping</span>
	<span className="green bold"> PASSED in 0h:0m:16s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 9 TESTS IN 0h:0m:16s<br/></span>
	<span className="blue bold">UP-TO-DATE: 8<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Как мы видим, команда ping отлично отработала, а это означает, что мы смогли полноценно настроить соединение между виртуальными машинами.

## Итоги

Итоговый тестовый сценарий можно скачать [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/7)