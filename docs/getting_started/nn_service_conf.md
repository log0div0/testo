#  Setting up testo-nn-server

## Settings file

All settings are store in a JSON file located here: `/etc/testo/nn_server.json`.

```json
{
	"port": 8156,
	"log_level": "info",
	"log_file": "/var/log/testo-nn-service.log",
	"license_path": "/opt/testo_license.lic",
	"use_gpu": false
}
```

Available settings:

`port` - Run te service on this TCP port

`log_level` - The logging level. By default it's "info", meaning only the essential messages are logged. You can also set the level to "trace", if you want to also log all kinds of additional info (including all JSON messages)

`log_file` - Path to the log file

`license_path` - Path to the license file. Applicable only for GPU mode (`use_gpu: true`)

`use_gpu` - Turn on/off GPU mode. Enabling the mode will require a valid license file (license_path)

> After changing the settings you have to restart `testo-nn-server` service

> By default the CPU object recognition mode is enabled (use_gpu: false). This mode is free and doesn't require a license. But this mode is rather slow and CPU-consuming. You can significantly accelerate the objects recognition (more than 10 times) by enabling the GPU mode. To get a license for that mode, please contact us support@testo-lang.ru

## GPU License request

### Installing latest Nvidia drivers

Before requesting the license it is require to install the latest possible Nvidia drivers on the machine with `testo-nn-server`


### Why would you need a license

In Testo Framework a license is needed to run `testo-nn-server` in GPU mode. In this mode the server enables GPU-acceleration (if there is any) to detect objects on screenshots. With this acceleration the objects detection is done several dozen times faster, so the overall time to run your test scenarios will be reduced significantly.

You don't need a license if you're content with default CPU detection mode.

### Getting a license request

To get a license - first you need to generate a request:

```text
testo_request_license [--out <path>]
```

-   `out <path>`: Save the generated license request to the specified path.

**Return value**: 0

> Running `testo_request_license` with no arguments will lead to saving the request to the `testo_license_request` file in the current working directory.

> The request must be generated on the machine where `testo-nn-server` service is located.

> If you see an error during the request generating - please make sure you have a GPU installed in your system, along with the latest drivers for it.

### Getting a license

At the moment the only way to get a license is with a personal request. Please contact us via email to learn the details: support@testo-lang.ru
