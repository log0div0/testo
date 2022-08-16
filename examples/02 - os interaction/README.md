# Testing app integration with the operating system

For this example, we created a toy-app named MySuperApp that lists all the users registered in the OS and, additionally, can add new users. We want to test that MySuperApp correctly interacts with the OS under various circumstances. We automated all these checks with a Testo Lang script.

## Directory overview

- `dist` - contains the pre-compiled MySuperApp package.
- `os_installation.testo` - contrains the preparatory tests, including the operating system installation.
- `tests.testo` - contains tests actually validating MySuperApp.

## What tests does this example include?

- `run_as_user` - checks MySuperApp when run by a regular user. A regular user doesn't have enough rights to do the user management, and MySuperApp must inform about that. But listing users should work. This test also checks that the MySuperApp user list updates after adding a new user.
- `run_as_admin` - checks MySuperApp when run by an admin. In this test we add a new user with MySuperApp and make sure that the new user is actually added in the system.

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
