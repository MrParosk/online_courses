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
