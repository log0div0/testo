# Virtual flash drive declaration

## Overview

> **Virtual flash drives are unavailable for Hyper-V**.

Virtual flash drives are mostly used for two purposes:
1. Transferring data between virtual machines.
2. Transeffring data between a virtual machine and a host.

You can insert a virtual flash drive to a virtual machine with actions [`(un)plug flash`](actions_vm#plug-flash) of the virtual machine commands. However, to transfer data between the Host and a virtual machine, you need the actions [`copyto`](actions_fd#copyto) and [`copyfrom`](actions_fd#copyfrom) of the flash drive commands.

There is also an opportunity to copy some data on the flash drive from the Host at the moment of this flash drive creation. It can be done with the optional `folder` attribute.

You can learn more about virtual flash drives in [this](/en/docs/tutorials/qemu/08_flash) tutorial.

## Declaration syntax

Declaration of a virtual flash drive starts with a `flash` keyword. The declaration looks like this:

```text
flash <name> {
  <attr1>: <value1>
  <attr2>: <value1>
  <attr3>: <value1>
  ...
}
```

> The declaration itself does not mean the actual creation of the flash drive. The actual creation happens when the first test mentioning this virtual flash drive is run.

> Virtual flash drives can also be defined inside macros. See [here](macros#macros-with-declarations) for more information.

Virtual Flash Drives, just like virtual machines, require unique entity-identifier `name`.

A virtual flash drive declaration is similar to a virtual machine [declaration](machine), but has a different set of attributes:

**Mandatory virtual flash drive attributes**:

- `size` - Type: memory size specifier or string. Flash drive size. If a string is used, the value inside the string must be convertible to a memory size literal. Inside the string [param referencing](param#param-referencing) is available.
- `fs` - Type: string. Filesystem type to format flash drive with. Possible values: `ntfs`, `fat`, `vfat`, `etx3`, `ext4`. Inside the string [param referencing](param#param-referencing) is available.

**Optional Virtual Flash Drive attributes**
- `folder` - Type: string. Path to a folder on the Host to copy on the flash drive right after its creation. Inside the string [param referencing](param#param-referencing) is available.

The `folder` attribute can be used to copy a folder from the Host to the flash drive after its creation. After the copying **the contents** of the folder will be placed in the root `/` directory on the flash. The folder itself isn't copied. You may consider `folder` as a mount-point for the virtual flash drive filesystem.

Configuration examples:

```testo
flash example_flash {
  fs: "ntfs"
  size: 32Mb
  folder: "./some_folder"
}
```

```testo
flash "${flash_name}" {
  fs: "ntfs"
  size: "${size_amount}"
  folder: "./some_folder"
}
```

In the second configuration the flash drive name depends on the `flash_name` param's value. The size of the flash drive depends on the `size_amount` param's value, which must be convertible to a memory size literal. Otherwise an error is generated.

> When running Testo under CentOS it is impossible by default to create virtual flash drives with the `ntfs` filesystem. It is because of the CentOS-distributed `libguestfs` library restrictions, which prevent formatting flash drives with the `ntfs` filesystem. You can bypass these restrictions with [compiling](https://www.redhat.com/archives/libguestfs/2016-February/msg00145.html) the `libguestfs` library from the source with NTFS enabled.

> Links are not allowed in the folder, specified in the `folder` attribute.

## Virtual flash drives caching

Virtual flash drives have a caching mechanism in Testo Framework, helping to check the integrity of their configuration. If the configuration has changed since the last Testo running (and therefore the cache is lost), then the flash drive must be re-created, and all the tests involving this flash drive must be re-run. This is one of the checks performed when [evaluating](test#validating-the-test-cache) the cache integrity.

The complete flash drives cache consistency checklist is this:

- Have the `fs`, `size` or `folder` attributes changed since the last successfull tests run?
- Have the `folder` contents changed since the last successful tests run?

If the answer to any of these questions is "yes" then the cache is lost and the flash drive must be re-created.

> The folder contents integrity is evaluated is follows. For each file inside the folder (and all the subfolders recursively) a checksum is calculated. The checksum could be calculated two ways. If the file's size is less than 1 MB (can be adjusted with the `content_cksum_maxsize` command-line argument) then the cksum is calculated based on the file's contents. Otherwise the cksum is calculated based on the timestamp of the last file's modification. You can change the threshold of the size for contents evaluation with the `content_cksum_maxsize` command line argument (1 MB by default). The resulting checksum is followed by the file's name. The files' checksums are summed together and thus the `folder` checksum is formed. If the cksum has changed since the last run - the consistency is broken and the flash drive must be re-created.
