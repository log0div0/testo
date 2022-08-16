# Testing installation and removing a program

For this example, we created a toy-app named MySuperApp and packed it inside an installer. We want to test that MySuperApp is installed and removed correctly. Additionally, we want to check that the installer successfully writes the necessary keys into the register. We automated all these checks with a Testo Lang script.

## Directory overview

- `dist` - contains the pre-compiled MySuperApp package.
- `os_installation.testo` - contrains the preparatory tests, including the operating system installation.
- `tests.testo` - contains tests actually validating MySuperApp.

## What tests does this example include?

- `install_my_super_app` - checks the MySuperApp installation correctness.
- `test_desktop_icon` - checks that MySuperApp correctly starts with the desktop icon. Therefore we make sure that the installator placed all the files where they belong.
- `test_context_menu` - checks that MySuperApp correctly starts with the context menu entry which should have been added after the MySuperApp installation. Therefore we make sure that the installator placed all the necessary keys into the Windows registry.
- `uninstall_my_super_app` - check the MySuperApp removing correctness.

## How can I run this example on my computer?

First, you need to download this repository on your computer:

```
git clone https://github.com/testo-lang/testo-examples.git
```

Second, you need to [download](https://testo-lang.ru/en/downloads) and install the Testo Framework. You will find instructions on getting started instructions [here](https://testo-lang.ru/en/docs/getting_started/getting_started)

Third, we've used the Windows 10 as the operating system to run MySuperApp. The Windows 10 installation is also a test (part of the tests hierarchy). Therefore you need to [download](https://www.microsoft.com/en-us/software-download/windows10ISO) the Windows 10 disk image. This example was tested with the Windows 10 May 2020 Update (build number 10.0.19041). For other Windows 10 versions the test scripts might need to be adjusted.

Congratulations! Now you're ready to run this example on your computer. Go to this example directory and run the following command:

```
sudo testo run ./tests.testo --param ISO_DIR /path/to/iso/dir
```

where the `/path/to/iso/dir` is the path to the directory containing the Windows 10 disk image.
