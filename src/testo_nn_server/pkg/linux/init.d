#!/bin/bash
# chkconfig: - 0 99

### BEGIN INIT INFO
# Provides: testo-nn-server
# Required-Start: $remote_fs $syslog
# Required-Stop: $remote_fs $syslog
# Default-Start:  2 3 4 5
# Default-Stop: 0 1 6
# Short-Description: Testo NN Server
# Description: Neural Networks server for Testo - a test automation framework
### END INIT INFO

testo-nn-server $@

exit $?
