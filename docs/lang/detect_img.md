# Detecting images on the screen

The Testo Framework allows you to detect images on the screen (as well as textlines) based on templates. A template is just a `png` picture with the image you want to detect (seel the example below).

The image detection is allowed in the actions `wait img`, `mouse click img` and the `if(check img)` statements.

## Preparing the template

Unlike the text detection, the image detection demands a little preparatory job to be done. This job is to create the expected image template.

The template creation is an easy process:

1. Take a screenshot of the virtual mahine's screen. The screenshot has to contain the image you want to detect in your test cases.
2. Open the screenshot in your favorite editor and cut the image you want to detect in the test cases.
3. Save the cut image on the disk as a png-image.

Of course, you can get the template image somewhere else (download from the Internet, for example). But the detection precision might be less than with the manually-created template.

Now, when the template is ready, you can specify the path to it in the actions `wait img`, `mouse click img` and the `if(check img)` statements.

## Examples

Let's consider the following situation. There is a screen:

<img src="/static/docs/lang/detect_img/desktop.png"/>

And in test cases we need to wait for three different components:

1. File Manager icon on the left dock panel;
2. "Some folder" folder;
3. "Another folder" folder.

Let's start with the File Manager icon. We take the screenshot of the VM and cut the file manager's icon:

<img src="/static/docs/lang/detect_img/selected_folder.png"/>

Then we save the icon in the file `/opt/icons/file_manager.png`.

Now all we need to do is specify the path to the saved template-picture:

```testo
wait img "/opt/icons/file_manager.png"
```

and we're all good to go for file manager icon detection.

Now let's deal with the folders detection. At this point it is important to choose the template wisely. If we created a too small template:

<img src="/static/docs/lang/detect_img/just_folder.png"/>

then instead of the specific folder we would search for all the folders (with the same icon):

<img src="/static/docs/lang/detect_img/2_imgs_detected.png"/>

To wait for the specific folder we need to extend the template and add the folder names to it:


<img src="/static/docs/lang/detect_img/some_folder.png"/>

<img src="/static/docs/lang/detect_img/another_folder.png"/>

Save the pictures to the files `/opt/icons/some_folder.png` and `/opt/icons/another_folder.png` respectively.

The search results are now different:

```testo
wait img "/opt/icons/some_folder.png"
```

<img src="/static/docs/lang/detect_img/some_folder_detected.png"/>

```testo
wait img "/opt/icons/another_folder.png"
```

<img src="/static/docs/lang/detect_img/another_folder_detected.png"/>
