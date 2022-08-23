# CI

`WIN10_TEMPLATE_PATH` is a manually prepared VM with the following software installed:

- VS 2019
- git
- cmake 3.19.2
- python 3.7.4 (must be install for all users, otherwise the guest additions would not be able to use it)
- wix 3.11

Apart from that it's nessesary to

- disable password prompt on startup
- disable Windows Defender
