# Lecture notes

## Lesson 1

### HTTP

HTTP is an application layer protocol that is sent over TCP. Due to its extensibility, it is used to not only fetch hypertext documents, but also images and videos or to post content to servers. HTTP can also be used to fetch parts of documents to update Web pages on demand. HTTP is stateless.

Clients and servers communicate by exchanging individual messages. The messages sent by the client, usually a Web browser, are called requests and the messages sent by the server as an answer are called responses.

### Stateless

Stateless means that different requests are independent.

The notion of statelessness is defined from the perspective of the server. The constraint says that the server should not remember the state of the application. As a consequence, the client should send all information necessary for execution along with each request, because the server cannot reuse information from previous requests as it didn't memorize them.

For example, if you're browsing an image gallery and the server has just send you image number 23, your client cannot simply say next to the server. Instead, it asks for image 24 to advance in the gallery. Indeed, your client has to supply all information necessary to execute the request, since the server does not remember that you were viewing image 23. The good thing is, you don't have to know either that you were viewing this particular image. Because, along with the representation of this image, the server can send you links labeled "previous" and "next", leading to the corresponding images.

It is vital to know that there are two kinds of state. There is application state, which is the kind of state we've talked about above, and resource state, which is the kind of state servers do deal with.

Application state is information about where you are in the interaction. It is used during your session with an application. For example, the fact that you are viewing picture 23, or the fact that you are logged in on Twitter, are both application state.

### Why Stateless

There are three important properties that are induced by statelessness:

1. Visibility
    - Every request contains all context necessary to understand it. Therefore, looking at a single request is sufficient to visualize the interaction.
2. Reliability
    - Since a request stands on its own, failure of one request does not influence others.
3. Scalability
    - The server does not have to remember the application state, enabling it to serve more requests in a shorter amount of time.

### REST (Representational State Transfer)

REST is a set of guidelines that leverage HTTP requests to transmit information. While SOAP is still used in some contexts, RESTful APIs currently dominate the API landscape. It is no longer an 'up and coming' technology, but rather simply the best practice for API development. In this course, we will be using JSON for content formatting.

### REST Constraints

REST applies a few additional specifications on top of HTTP.

1. Separation of client and server
2. Stateless
    - Stateful server remembers client's activity between requests.
    - RESTful architecture does not allow retention of client information between requests. Each request is independent.
    - Tokens provide some memory functionality in RESTful architectures.
3. Cacheable
4. Uniform interface
5. Layered system
6. Code on demand

## Lesson 2

### Structure of an HTTP request

- Header
  - Request line
    - HTTP verb
    - URI
    - HTTP/version
  - Optional request headers
    - name:value, name:value
- Blank line
- Body (optional)
  - Additional information

#### Example HTTP request

```text
GET puppies.html HTTP/1.1
Host: www.puppyshelter.com
Accept: image/gif, image/jpeg, */*
Accept-Language: en-us
Accept-Encoding: gzip, deflate
User-Agent: Mozilla/4.0
Content-Length: 35

puppyId=12345&name=Fido+Simpson
```

### Structure of an HTTP response

- Header
  - Status line
    - HTTP/version
    - Status code
    - Reason phrase
  - Optional response headers
- Blank line
- Body (optional)

#### Example HTTP response

```text
HTTP/1.1 OK
Date: Fri, 04 Sep 2015 01:11:12 GMT
Server: Apache/1.3.29 (Win32)
Last-Modified: Sat, 07 Feb 2014
ETag: "0-23-4024c3a5:
ContentType: text/html
ContentLength: 35
Connection: KeepAlive
KeepAlive: timeout=15, max = 100

<h1>Welcome!</h1>
```

### URL versus URI

- URIs are identifiers, and that can mean name, location, or both.
- The part that makes something a URL is the combination of the name and an access method, such as https://.

### cURL

Curl is a popular command line tool for sending and receiving HTTP. At its very basic, cURL makes requests to URL's. Normally you want to interact with those URL's in someway. By default, cURL makes an HTTP GET request.

Some useful curl commands:

- -i Includes the header
  - `curl -i https://jsonplaceholder.typicode.com/posts`
- --header Returns only the header
  - `curl --header https://jsonplaceholder.typicode.com/posts`
- -o Saves the output to a given file
  - `curl -o example.txt https://jsonplaceholder.typicode.com/posts`
- -O Download the file (directly)
  - `curl -O https://jsonplaceholder.typicode.com/posts`
- -X Specifies a custom request method instead of GET (e.g. PUT, DELETE etc)
  - `curl -X DELETE https://jsonplaceholder.typicode.com/posts/1`
- -d Sends the specified data in a POST request
  - `curl -d test.txt https://jsonplaceholder.typicode.com/posts`
- -u allows for authentication
  - `curl -u username:password https://jsonplaceholder.typicode.com/posts`

The official cURL documentation can be found [here](http://curl.haxx.se/docs/manpage.html).

### HTTP request methods

- GET
  - GET requests a representation of the specified resource. Requests using GET should only retrieve data.
- POST
  - The POST method is used to submit an entity to the specified resource, often causing a change in state or side effects on the server.
- PUT
  - The PUT method replaces all current representations of the target resource with the request payload.
- DELETE
  - The DELETE method deletes the specified resource.
- PATCH
  - The PATCH method is used to apply partial modifications to a resource.

## Lesson 4

### Storing user password

Hashing is a one-way function which allows us to store user password in our database without jeopardizing the user's security if the database were hacked. It is recommended to login over HTTPS (secure HTTP).

### Token Based Authentication

A token is a string that the server generates. The token can be passed along with the HTTP request, like cookies, but tokens don't depend on a browser. This allows the user to authenticate itself without using their username and password, making it safer. Tokens are useful since they typically have an expiration time, making it less troublesome if the token were leaked since it will expire.

It's typical to use a cryptographically signed tokens, making it harder to tamper with the token. One library that does this in Python is `itsdangerous`.

### OAuth 2.0

OAuth allows an application to access data or services that the user has with another service, and this is done in a way that prevents the application from knowing the login credentials that the user has with the service.

For example, consider a website or application that asks you for permission to access your Facebook account and post something to your timeline. In this example you are the resource holder (you own your Facebook timeline), the third party application is the consumer and Facebook is the provider. Even if you grant access and the consumer application writes to your timeline, it never sees your Facebook login information.

## Lesson 5

### Developer Friendly Documentation

> Documentation should be easy to navigate and aesthetically pleasing.

### Using Proper URIs

- URIs should always refer to resources, not actions being performed on those resources.
- Use plural for each resource name.

### Versioning Your API

API version can be specified in the URI. For example:

```text
GET /api/v1/puppies
GET /api/v2/puppies
```
