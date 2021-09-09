#!/bin/bash
# chkconfig: - 0 99

### BEGIN INIT INFO
# Provides: testo-guest-additions
# Required-Start: $remote_fs $syslog
# Required-Stop: $remote_fs $syslog
# Default-Start:  2 3 4 5
# Default-Stop: 0 1 6
# Short-Description: Testo Guest Additions
# Description: Guest additions for Testo - a test automation framework
### END INIT INFO

testo-nn-service $@

exit $?
