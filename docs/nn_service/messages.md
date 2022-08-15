# Interaction with Testo NN Service

## Overview

Client-service interaction may be described as followed:

1. The client generates a request which is basically a JSON file. This request should contain:
	- The screenshot of interest in the form of a PNG-image
	- Text with JavaScript-snippet describing "What should be found"
2. Generated request should be serialized in CBOR format
3. The client builds up a message for the service. A message consists of a header and the CBOR-serialized JSON that was created during step 2. The resulting message is to be sent to the service.
4. The service receives the message and processes the request with built-in NN engine and JavaScript interpreter.
5. (Applicablt only when searching for images) Optinally the service can request additional info from the client (see below).
6. The service builds up and sends the response message back to the client. The message contains a header and JSON-body serizalized in CBOR format. The JSON contains all the information about found images and their location.

## Message structure

<img src="/static/docs/nn_service/Message_en.svg"/>

All messages in the client-service interaction consist of a header and a body.

**A Header** is just one 4-bytes field. This field must containt the body size in bytes.

**A Body** is basically JSON-file serialized in CBOR format. Exact fields set depends on the type of the message.

## Messages from the client to the service

## JavaScript evaluation request (JSRequest)

<img src="/static/docs/nn_service/JSRequest_en.svg"/>

That's main request message to the service. With this message you can pass a screenshot to the service and "ask" it to do some work with it: find a text or an image on it (or a combination of several textlines and images).

**Fields**

- `version` - Type: string. Protocol version number. At the moment the only possible value is `"1.0"`
- `type` - Type: string. The type of the message. Must be `js_eval`
- `image` - Type: binary. Screenshot in PNG format
- `js_script` - Type: string. JavaScript-snippet which is to be interpreted for the screenshot.

### JavaScript-snippets guidelines

All the logic regarding looking for objects on screenshots is represented with JavaScript snippets. You can look up the main concepts in the Testo Lang [JS-selectors](/en/docs/js/general) guide.

Inside a JavaScript-snippet you can use global functions `find_text` (find a textline) and `find_img` (find an image). You can aso combine them in any fashion you like.

> Keep in mind that when processing a `find_img` call, the service will request additional info about the sought-for image from the client.

JavaScript-snippet must end with a `return` statement for the object you'd like to return. When processing a JS-snippet, the service converts the returned object into JSON-document (if such conversion is possible) and sends the resulting JSON to the client in the response message.

Here are some examples of objects you can return from JavaScript-snippets:
- TextTensor (a set of rectangles containing textlines). Is converted to a JSON-array of "Rectangle" objects;
- ImgTensor (a set of rectangles containing images). Is converted to a JSON-array of "Rectangle" objects;
- Poing (a Point is a result of many methods of `TextTensor` and `ImgTensor`. For instance - `center`). Is converted to a JSON-object.
- boolean (true or false).

### JSRequests with `find_img` calls

The Service acts a bit differently when proccessing JavaScript-snippets with `find_img` calls: for each such call the service will send a `RefImageRequest` to the client, asking to provide the sought-for image, the path to which is specified as an argument to `find_img` call. The client should be ready for accepting `RefImageRequst` messages from the service and providing the sought-for images to the service in `RefImage` messages.

## Message with sought-for images for `find_img` calls (RefImage)

<img src="/static/docs/nn_service/RefImage_en.svg"/>

The message that should be sent back to the service in response to `RefImageRequest` message. Must contain the sought-for image in PNG-format, the path to which was specified in the corresponding `RefImageRequest` message.

**Fields**

- `version` - Type: string. Protocol version number. At the moment the only possible value is `"1.0"`
- `type` - Type: string. The type of the message. Must be `ref_image`
- `image` - Type: binary. The sought-for image (template) in PNG format

## Messages from the service to the client

## JS-snippet proccessing results (JSEvalResult)

<img src="/static/docs/nn_service/JSEvalResult_en.svg"/>

This is the message with JS-requests interpreting results. Contains the JSON-object that was returned in `return` statement of the JS-snippet. The JSON fields depend on the type of the returned object.

**Fields**

- `version` - Type: string. Protocol version number. At the moment the only possible value is `"1.0"`
- `type` - Type: string. The type of the message. Must be `eval_result`
- `data` - Type: JSON-object. The result of JS-snippet proccessing (the returned object)
- `stdout` - Type: string. Contains stdout that might have been generated during the JS-snippet proccessing. Could be an empty string.

## Request for the sought-for image (RefImageRequest)

<img src="/static/docs/nn_service/RefImageRequest_en.svg"/>

This request is generated by the service every time it proccesses `find_img(path_to_template)` call. Since the service doesn't have the access to the `template` image (there is a possiblitiy that the client and the service reside on different machines), it has to request the additional info (the template in PNG format) from the client.

> The number of requests is exactly the same as the number of times a `find_img` function is called. If the snippet contains 2 calls for `find_img` - then the service will generate two independent `RefImageRequest` messages.

**Fields**

- `version` - Type: string. Protocol version number. At the moment the only possible value is `"1.0"`
- `type` - Type: string. The type of the message. Must be `ref_image_request`
- `data` - Type: string. The path to the image that the client has to provide. Is equal to the `path_to_template` argument value passed to the `find_img` call.


## Error message

<img src="/static/docs/nn_service/Error_en.svg"/>

This message is generated by the service in a case of any emergency situation. It's also generated by the client when respoindig to `RefImageRequest` in the case when it's impossible to provide the requested sought-for image.

**Fields**

- `version` - Type: string. Protocol version number. At the moment the only possible value is `"1.0"`
- `type` - Type: string. The type of the message. Must be `error`
- `data` - Type: string. The error message
- `stdout` - Type: string. Contains stdout that might have been generated during the JS-snippet proccessing. Could be an empty string.

## Examples

### Find "Hello world" text on the screenshot

**Send the request to the service**

1. Generate the JS-snippet to find the textline:

```js
    return find_text('Hello world')
```

2. Generate the JSRequest message body:

```json
{
    "image": "omitted",
    "js_script": "return find_text('Hello world')",
    "type": "js_eval",
    "version": "1.0"
}
```

> The field `image` should contain the binary PNG-screenshot instead of "omitted".

3. Serialize the message into CBOR format.
4. Add the header to the message - 4 bytes containing the size of the resulting JSON-document serialized in CBOR.
5. Send the message to the server.

**Receive the response**

1. Receive the header of the message from the service (4 bytes). The value in this header will tell you how large the message body in bytes.
2. Receive the message body and deserialize it from CBOR to get this JSON:

```json
{
    "version": "1.0",
    "type": "eval_result",
    "stdout": "",
    "data": [
        {
            "bottom": 409,
            "left": 409,
            "right": 493,
            "top": 389
        },
        {
            "bottom": 478,
            "left": 1271,
            "right": 1332,
            "top": 466
        },
        {
            "bottom": 622,
            "left": 1012,
            "right": 1098,
            "top": 601
        }
    ]
}
```

3. Check the message type (should not be `error`) and go to the `data` field where we can see the result of the `find_text` function. In our case it's a list of rectangles containing all the locations of all found instances of "Hello world".

## Find the youtube icon (youtube.png) on the screenshot

**Send the request to the service**

1. JS-snippet to commence the search:

```js
    return find_img('img/youtube.png')
```

2. Build the JSRequest body

```json
{
    "image": "omitted",
    "js_script": "return find_img('img/youtube.png')",
    "type": "js_eval",
    "version": "1.0"
}
```

> The field `image` should contain the binary PNG-screenshot instead of "omitted".

3. Serialize the message to CBOR format
4. Add the header - 4 bytes containing the size of screenshot.
5. Send the message to the service.

**Wait for the RefImageRequest from the service**

Since the JS-snippet containt a `find_img` call, the client has to anticipate an additional request about the sought-for picture from the service.

Receive the additional message as usual:

```json
{
    "version": "1.0",
    "type": "ref_image_request",
    "data": "img/youtube.png"
}
```

**Send the RefImage message**

Build the RefImage body:

```json
{
    "image": "omitted",
    "type": "ref_image",
    "version": "1.0"
}
```

> The field `image` should contain the binary PNG-screenshot `img/youtube.png` instead of "omitted".

Attach the header and send the message as usual.

**Receive the result from the service**

```json
{
    "version": "1.0",
    "type": "eval_result",
    "stdout": "",
    "data": [
        {
            "bottom": 516,
            "left": 222,
            "right": 493,
            "top": 389
        }
    ]
}
```

### Complex request

Example: find the text "Hello world" and two icons: "youtube.png" and "facebook.png". If all three elements are present in single instance - then return the center of "Hello world". Instead - return false.

**Send the request to the service**

JS-snippet:

```js
    let text = find_text('Hello world')
    let youtube = find_img('img/youtube.png')
    let facebook = find_img('img/facebook.png')

    if (text.size() == 1 && youtube.size() == 1 && facebook.size() == 1) {
        print('Found!')
        return text.center()
    } else {
        print('Not found!')
        return false
    }
```

**Wait for 2 RefImageRequest messages**

*for youtube.png*

```json
{
    "version": "1.0",
    "type": "ref_image_request",
    "data": "img/youtube.png"
}
```

Response from the client:

```json
{
    "image": "omitted",
    "type": "ref_image",
    "version": "1.0"
}
```

> The field `image` should contain the binary PNG-screenshot `img/youtube.png` instead of "omitted".

*for facebook.png*

```json
{
    "version": "1.0",
    "type": "ref_image_request",
    "data": "img/facebook.png"
}
```

Response from the client:

```json
{
    "image": "omitted",
    "type": "ref_image",
    "version": "1.0"
}
```

> The field `image` should contain the binary PNG-screenshot `img/facebook.png` instead of "omitted".

**Receive the result**

*If the condition is met*

```json
{
    "version": "1.0",
    "type": "eval_result",
    "stdout": "Found",
    "data": {
       "x": 409,
       "y": 522
    }
}
```

*Otherwise*

```json
{
    "version": "1.0",
    "type": "eval_result",
    "stdout": "Not found!",
    "data": false
}
```