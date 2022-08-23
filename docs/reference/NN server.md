#  Setting up testo-nn-server

## Config file

All settings are store in a JSON file located here: `/etc/testo/nn_server.json`. The default content of this file:

```json
{
	"port": 8156,
	"log_level": "info",
	"log_file": "/var/log/testo-nn-service.log",
	"use_gpu": false
}
```

Available settings:

- `port` - Run te service on this TCP port
- `log_level` - The logging level. By default it's `info`, meaning only the essential messages are logged. You can also set the level to `trace`, if you want to also log all kinds of additional info (including all protocol messages)
- `log_file` - Path to the log file
- `use_gpu` - Turn on/off GPU mode

> After changing the settings you have to restart `testo-nn-server` service
