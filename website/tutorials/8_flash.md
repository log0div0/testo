# Часть 8. Флешки

## С чем Вы познакомитесь

В этой части вы познакомитесь с механизмом виртуальных флешек в платформе Testo.

## Начальные условия

1. Платформа `testo` установлена.
2. Установлен менеджер виртуальных машин `virt-manager`.
3. На хостовой машине имеется прямой (без прокси) доступ в Интернет
4. Имеется установочный образ [Ubuntu server 16.04](http://releases.ubuntu.com/16.04/ubuntu-16.04.6-server-amd64.iso) с расположением `/opt/iso/ubuntu_server.iso`. Местоположение и название установочного файла может быть другим, в этом случае нужно будет соответствующим образом поправить тестовый скрипт.
5. Имеется образ с гостевыми дополнениями Testo в одной папке с установочным образом Ubuntu.
6. (Рекомендовано) Настроена [подсветка синтаксиса](/docs/getting_started/getting_started#настройка-подсветки-языка-testo-lang) `testo-lang` в Sublime Text 3.
7. (Рекомендовано) Проделаны шаги из [седьмой части](7_nat).

## Вступление

Помимо виртуальных сетей и виртуальных машин в платформе Testo есть ещё один вид виртуальных сущностей - виртуальные флешки. Виртуальные флешки могут использоваться в двух случаях:

1. Для передачи файлов между виртуальными машинами, не используя для этого сеть;
2. Загрузка файлов в виртуальную машину из хоста если нет возможности использовать гостевые дополнения и действие `copyto`.

При этом для флешек в платформе Testo есть два важных свойства:

1. При успешном завершении теста, в кототом участвовала флешка, для этой флешки фиксируется её состояние (как и для виртуальных машин), так что можно быть уверенным, что в ходе выполнения теста флешка всегда находится в правильном состоянии;
2. Атрибуты флешки (в том числе целостность файлов, которые передаются внутрь гостевой системы через флешку) участвуют в подсчете целостности теста, в котором было обращение к флешке. То есть, как и в случае с действием `copyto`, вы можете быть уверены, что платформа Testo отследит изменение файлов, участвующих в тесте, и запустит тест заново при необходимости.

Есть также и важное **ограничение** в использовании флешек в платформе Testo: если флешка была подключена к виртуальной машине в ходе теста, то до окончания теста эту флешку **необходимо отключить**.

Также на текущий момент не допускается одновременное подключение двух и более флешек в одну виртуальную машину.

В этом уроке мы познакомимся с виртуальными флешками и всеми особенностями их использования.

## С чего начать?

Давайте представим, что нам необходимо передать файл из машины `client` в машину `server` и по какой-то причине мы не хотим пользоваться передачей файлов по сети. В этой ситуации нам как раз может помочь виртуальная флешка. Для того, чтобы объявить виртуальную флешку, используется директива [`flash`](/docs/lang/flash).

```testo
flash exchange_flash {
	fs: "ntfs"
	size: 16Mb
}
```

Флешки объявляются схожим образом с виртуальными машинами и сетями: директива `flash`, уникальное среди флешек имя, а также набор атрибутов, часть которых являются обязательными. Для флешек обязательных атрибута два:

1. `fs` - тип файловой системы. В нашем случае это `ntfs`;
2. `size` - размер флешки. Нам будет более чем достаточно 16 Мегабайт.

Как и в случае с виртуальными машинами, объявление флешки само по себе не приводит к её созданию. Флешка создаётся только при запуске теста, в котором эта флешка фигурирует.

Давайте попробуем сделать новый тест, в котором наша флешка будет использоваться для передачи файлов между машинами.

```testo
#обратите внимание, тест отнаследован не от test_ping, 
#а тех же client_prepare и server_prepare
#То есть test_ping и exchange_files_with_flash лежат на одном уровне
#иерархии в дереве тестов
test exchange_files_with_flash: client_prepare, server_prepare {
	client {
		#Создаём файл, который нужно будет передать на сервер
		exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"

		plug flash exchange_flash
		sleep 5s
		exec bash """
			mount /dev/sdb1 /media
			cp /tmp/copy_me_to_server.txt /media
			umount /media
		"""

		unplug flash exchange_flash
	}

	server {
		plug flash exchange_flash
		sleep 5s
		exec bash """
			mount /dev/sdb1 /media
			cp /media/copy_me_to_server.txt /tmp
			umount /media
			cat /tmp/copy_me_to_server.txt
		"""

		unplug flash exchange_flash
	}
}
```

В целом, тест достаточно прямолинеен: на клиенте мы создаем файл, который необходимо передать на сервер, затем выполняем действие `plug flash`, которое подключает новое устройство к виртуальной машине (то же самое происходит когда человек вставляет флешку в компьютер).

Т.к. в Ubuntu Server не предусмотрено автоматическое монтирование флешки в файловую систему, нам необходимо сделать это самостоятельно. Для начала необходимо выждать несколько секунд, чтобы ОС успела отреагировать на новое подключенное устройство вставленная флешка внутри виртуалки будет видна как устройство `/dev/sdb`, от которого нам требуется первый раздел (то есть устройство `/dev/sdb1`). Монтируем этот раздел, копируем на флешку файл, отмонтируем устройство и безопасно извлекаем флешку.

На сервере мы снова проделываем те же манипуляции и выводим содержимое переданного файла. В конце теста не забываем извлечь флешку из сервера.

Попробуем все это запустить (на этот момент все предыдущие тесты должны быть закешированы чтобы получился такой вывод)

<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso --test_spec exchange_files_with_flash<br/></span>
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
	<span className="magenta ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 89%] Preparing the environment for test </span>
	<span className="yellow ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 89%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 89%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 89%] Creating flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="">Warning: The resulting partition is not properly aligned for best performance.<br/></span>
	<span className="">The partition start sector was not specified for /dev/nbd0p1 and it could not be obtained automatically.  It has been set to 0.<br/></span>
	<span className="">The number of sectors per track was not specified for /dev/nbd0p1 and it could not be obtained automatically.  It has been set to 0.<br/></span>
	<span className="">The number of heads was not specified for /dev/nbd0p1 and it could not be obtained automatically.  It has been set to 0.<br/></span>
	<span className="">Cluster size has been automatically set to 4096 bytes.<br/></span>
	<span className="">To boot from a device, Windows needs the 'partition start sector', the 'sectors per track' and the 'number of heads' to be set.<br/></span>
	<span className="">Windows will not be able to boot from this device.<br/></span>
	<span className="">Creating NTFS volume structures.<br/></span>
	<span className="">mkntfs completed successfully. Have a nice day.<br/></span>
	<span className="">/dev/nbd0 disconnected<br/></span>
	<span className="blue ">[ 89%] Taking snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 89%] Running test </span>
	<span className="yellow ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 89%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ echo 'Hello from client!'<br/></span>
	<span className="blue ">[ 89%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 89%] Sleeping in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 89%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /tmp/copy_me_to_server.txt /media<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className="blue ">[ 89%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 89%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 89%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 89%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /media/copy_me_to_server.txt /tmp<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className=" ">+ cat /tmp/copy_me_to_server.txt<br/></span>
	<span className=" ">Hello from client!<br/></span>
	<span className="blue ">[ 89%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 89%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 89%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 89%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">exchange_files_with_flash</span>
	<span className="green bold"> PASSED in 0h:0m:19s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 9 TESTS IN 0h:0m:19s<br/></span>
	<span className="blue bold">UP-TO-DATE: 8<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 1<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Как мы видим, перед началом теста была создана виртуальная флешка, которая затем успешно использовалась для передачи файлов между машинами. В конце мы видим приветствие от клиента, которое сигнализирует о том, что файл был успешно передан.

Также еще раз необходимо напомнить, что в конце теста состояние флешки фиксируется, и если в более поздних тестах на эту флешку будут записаны новые файлы, они не будут проявляться в более ранних тестах.

Еще раз остановимся на замечании, что все флешки необходимо вытащить из виртуальных машин перед завершением теста. Если это условие не будет солблюдено, при фиксации результатов теста будет получена ошибка.

## Копирование файлов из хоста с помощью флешки

Давайте рассмотрим второй сценарий использования виртуальных флешек - передача файлов с хоста на виртуальные машины при невозможности/нежелательности использования гостевых дополнений.

Для этого в объявлении флешки необходимо указать ещё один атрибут `folder`.

```testo
flash exchange_flash {
	fs: "ntfs"
	size: 16Mb
	folder: "./folder_to_copy"
}
```

В этом атрибуте необходимо указать путь к папке **на хосте**, которую необходимо скопировать на флешку. Можно использовать относительный путь, в этом случае он будет начинаться с того места. где расположен сам файл с тестовыми сценариями. В качестве `folder` нельзя указывать один файл, **это обязательно** должна быть папка.

Давайте представим, что по какой то причине мы не можем поспользоваться действием `copyto` в виртуальной машине `client` и  не можем передать скрипт `rename_net.sh` для переименования сетевых интерфейсов внутри виртуальной машины. В этом случае мы можем воспользоваться флешкой.

Создайте папку `folder_to_copy` в той же папке, что и файл `hello_world.testo` и скопируйте файл `rename_net.sh` внутрь `folder_to_copy`. Теперь надо лишь немного подправить тест `client_prepare`.

```testo
test client_prepare: client_unplug_nat {
	client {
		plug flash exchange_flash
		sleep 5s
		exec bash """
			mount /dev/sdb1 /media
			cp /media/rename_net.sh /opt/rename_net.sh
			umount /media
		"""
		unplug flash exchange_flash
		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:aa server_side
			ip a a 192.168.1.2/24 dev server_side
			ip l s server_side up
			ip ad
		"""
	}
}
```

Попробуем запустить все тесты

<Terminal height="600px">
	<span className="">user$ sudo testo run ~/testo/hello_world.testo --stop_on_fail --param ISO_DIR /opt/iso<br/></span>
	<span className="">Some tests have lost their cache:<br/></span>
	<span className="">	- client_prepare<br/></span>
	<span className="">Do you confirm running them and all their children? [y/N]: y<br/></span>
	<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
	<span className="magenta ">server_install_ubuntu<br/></span>
	<span className="magenta ">server_install_guest_additions<br/></span>
	<span className="magenta ">server_unplug_nat<br/></span>
	<span className="magenta ">server_prepare<br/></span>
	<span className="magenta ">client_install_ubuntu<br/></span>
	<span className="magenta ">client_install_guest_additions<br/></span>
	<span className="magenta ">client_unplug_nat<br/></span>
	<span className="blue bold">TESTS TO RUN:<br/></span>
	<span className="magenta ">client_prepare<br/></span>
	<span className="magenta ">test_ping<br/></span>
	<span className="magenta ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 70%] Preparing the environment for test </span>
	<span className="yellow ">client_prepare<br/></span>
	<span className="blue ">[ 70%] Restoring snapshot </span>
	<span className="yellow ">client_unplug_nat</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 70%] Creating flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="">Warning: The resulting partition is not properly aligned for best performance.<br/></span>
	<span className="">The partition start sector was not specified for /dev/nbd0p1 and it could not be obtained automatically.  It has been set to 0.<br/></span>
	<span className="">The number of sectors per track was not specified for /dev/nbd0p1 and it could not be obtained automatically.  It has been set to 0.<br/></span>
	<span className="">The number of heads was not specified for /dev/nbd0p1 and it could not be obtained automatically.  It has been set to 0.<br/></span>
	<span className="">Cluster size has been automatically set to 4096 bytes.<br/></span>
	<span className="">To boot from a device, Windows needs the 'partition start sector', the 'sectors per track' and the 'number of heads' to be set.<br/></span>
	<span className="">Windows will not be able to boot from this device.<br/></span>
	<span className="">Creating NTFS volume structures.<br/></span>
	<span className="">mkntfs completed successfully. Have a nice day.<br/></span>
	<span className="">/dev/nbd0 disconnected<br/></span>
	<span className="">/dev/nbd0 disconnected<br/></span>
	<span className="blue ">[ 70%] Taking snapshot </span>
	<span className="yellow ">initial</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 70%] Running test </span>
	<span className="yellow ">client_prepare<br/></span>
	<span className="blue ">[ 70%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 70%] Sleeping in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 70%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /media/rename_net.sh /opt/rename_net.sh<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className="blue ">[ 70%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 70%] Executing bash command in virtual machine </span>
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
	<span className="blue ">[ 70%] Taking snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 70%] Taking snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="green bold">[ 80%] Test </span>
	<span className="yellow bold">client_prepare</span>
	<span className="green bold"> PASSED in 0h:0m:9s<br/></span>
	<span className="blue ">[ 80%] Preparing the environment for test </span>
	<span className="yellow ">test_ping<br/></span>
	<span className="blue ">[ 80%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 80%] Running test </span>
	<span className="yellow ">test_ping<br/></span>
	<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ ping 192.168.1.2 -c5<br/></span>
	<span className=" ">PING 192.168.1.2 (192.168.1.2) 56(84) bytes of data.<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=1 ttl=64 time=0.041 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=2 ttl=64 time=0.032 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=3 ttl=64 time=0.044 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=4 ttl=64 time=0.043 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.2: icmp_seq=5 ttl=64 time=0.043 ms<br/></span>
	<span className=" "><br/></span>
	<span className=" ">--- 192.168.1.2 ping statistics ---<br/></span>
	<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 3998ms<br/></span>
	<span className=" ">rtt min/avg/max/mdev = 0.032/0.040/0.044/0.008 ms<br/></span>
	<span className="blue ">[ 80%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ ping 192.168.1.1 -c5<br/></span>
	<span className=" ">PING 192.168.1.1 (192.168.1.1) 56(84) bytes of data.<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.055 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.039 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.043 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=4 ttl=64 time=0.045 ms<br/></span>
	<span className=" ">64 bytes from 192.168.1.1: icmp_seq=5 ttl=64 time=0.044 ms<br/></span>
	<span className=" "><br/></span>
	<span className=" ">--- 192.168.1.1 ping statistics ---<br/></span>
	<span className=" ">5 packets transmitted, 5 received, 0% packet loss, time 4000ms<br/></span>
	<span className=" ">rtt min/avg/max/mdev = 0.039/0.045/0.055/0.006 ms<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 80%] Taking snapshot </span>
	<span className="yellow ">test_ping</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[ 90%] Test </span>
	<span className="yellow bold">test_ping</span>
	<span className="green bold"> PASSED in 0h:0m:14s<br/></span>
	<span className="blue ">[ 90%] Preparing the environment for test </span>
	<span className="yellow ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 90%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 90%] Restoring snapshot </span>
	<span className="yellow ">client_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Restoring snapshot </span>
	<span className="yellow ">server_prepare</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 90%] Running test </span>
	<span className="yellow ">exchange_files_with_flash<br/></span>
	<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ echo 'Hello from client!'<br/></span>
	<span className="blue ">[ 90%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Sleeping in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
	<span className="yellow ">client</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /tmp/copy_me_to_server.txt /media<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className="blue ">[ 90%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Plugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">into virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 90%] Sleeping in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> for 5s<br/></span>
	<span className="blue ">[ 90%] Executing bash command in virtual machine </span>
	<span className="yellow ">server</span>
	<span className="blue "> with timeout 10m<br/></span>
	<span className=" ">+ mount /dev/sdb1 /media<br/></span>
	<span className=" ">+ cp /media/copy_me_to_server.txt /tmp<br/></span>
	<span className=" ">+ umount /media<br/></span>
	<span className=" ">+ cat /tmp/copy_me_to_server.txt<br/></span>
	<span className=" ">Hello from client!<br/></span>
	<span className="blue ">[ 90%] Unplugging flash drive </span>
	<span className="yellow ">exchange_flash </span>
	<span className="blue ">from virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="blue ">[ 90%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for flash drive </span>
	<span className="yellow ">exchange_flash<br/></span>
	<span className="blue ">[ 90%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">client<br/></span>
	<span className="blue ">[ 90%] Taking snapshot </span>
	<span className="yellow ">exchange_files_with_flash</span>
	<span className="blue "> for virtual machine </span>
	<span className="yellow ">server<br/></span>
	<span className="green bold">[100%] Test </span>
	<span className="yellow bold">exchange_files_with_flash</span>
	<span className="green bold"> PASSED in 0h:0m:19s<br/></span>
	<span className="blue bold">PROCESSED TOTAL 10 TESTS IN 0h:0m:43s<br/></span>
	<span className="blue bold">UP-TO-DATE: 7<br/></span>
	<span className="green bold">RUN SUCCESSFULLY: 3<br/></span>
	<span className="red bold">FAILED: 0<br/></span>
	<span className="">user$ </span>
</Terminal>

Как мы видим, все тесты успешно отработали, так что передача `rename_net.sh` внутрь виртуальной машины успешно состоялась.

Можно обратить внимание, что флешка была создана заново. Это произошло только потому, что мы имзенили её конфигурацию (добавили атрибут `folder`). Если бы целостность конфигурации осталась нетронутой, был бы восстановлен снепшот `initial`.

Также можно заметить, что тест `test_ping` был проведен заново, хотя мы его вовсе не трогали. Это произошло потому что мы изменили тест `client_prepare` - предшественника теста `test_ping`.

Как уже упоминалось ранее, целостность файлов, указанных в `folder`, участвует в целостности теста, в котором участвует флешка. Если поменять содержимое папки `folder_to_copy` (в том числе просто поменять файл `rename_net.sh`), то тест `client_prepare` будет запущен заново. Предлагаем вам убедиться в этом самостоятельно.

## Итоги

Виртуальные флешки - последний тип виртуальных сущностей, которые есть в Testo (наряду с виртуальными машинами и сетями). Их можно использовать как для передачи файлов между машинами, так и для передачи файлов с хоста на виртуальную машину в условиях недоступности или нежелательности гостевых дополнений.

Дерево тестов на текущий момент выглядит следующим образом

![Tests hierarchy](/static/tutorials/8_flash/test_hierarchy.png)

Готовый скрипт можно найти [здесь](https://github.com/CIDJEY/Testo_tutorials/tree/master/8)