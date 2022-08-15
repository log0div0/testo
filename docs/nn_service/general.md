# Overview

The Testo Framework consists of two major components: Testo-nalg interpreter called `testo`  and objects finding service called `testo-nn-server`. It is with the help of this service `testo` interpreter "learns" about presence/absence of certain textlines and images on screenshots.

<img src="/static/docs/nn_service/Protocol_en.svg"/>

But the service `testo-nn-server` can be quite helpful even by itself - without the `testo` interpreter. If you have some screenshots and you would like to get the information about certain objects on them (and the coordinates of these objects) - then you can build your own client application interacting with `testo-nn-server` and use that information for your own need.

In this chapther you can find the `testo-nn-server` interaction protocol and the format of the messages used in this protocol. By implementing this protocol, you will be able to interact with `testo-nn-server` from your own application.
