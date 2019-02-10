# How to write clean code

## Meaningful names

Use intention revealing names. Choosing good names takes time but saves more than it takes.The name of a variable, function, or class, should answer all the big questions. It should tell you why it exists, what it does, and how it is used. If a name requires a comment, then the name does not reveal its intent. For example:

```python
    i = 1 # elapse time in days.
```

We should choose a name that specifies what is being measured and the unit of that measurement.
A better name would be elapsed_time.

### Class names

Classes and objects should have noun or noun phrase names like Customer, WikiPage, Account, and AddressParser. A class name should not be a verb.

### Method names

Methods should have verb or verb phrase names like post_payment, delete_page, or save.

### Pick one word per concept

Pick one word for one abstract concept and stick with it. For instance, it's confusing to have fetch, retrieve, and get as equivalent methods of different classes. How do you remember which method name goes with which class?

## Functions

The first rule of functions is that they should be small. This implies that the blocks within if statements, else statements, while statements, and so on should be one line long. Not only does this keep the enclosing function small, but it also adds documentary value because the function called within the block can have a nicely descriptive name.

### Function arguments

A function shouldn't have more than 3 arguments. Keep it as low as possible. When a function seems to need more than two or three arguments, it is likely that some of those arguments ought to be wrapped into a class of their own.

## Comments

If you are writing comments to prove your point, you are doing a blunder. Ideally, comments are not required at all. If your code needs commenting, you are doing something wrong. Our code should explain everything. Modern programming languages are English like through which we can easily explain our point. Correct naming can prevent comments.

# SOLID

## S — Single Responsibility Principle

`A class should have one, and only one, reason to change.`

When requirements change, this implies that the code has to undergo some reconstruction, meaning that the classes have to be modified. The more responsibilities a class has, the more change requests it will get, and the harder those changes will be to implement. The responsibilities of a class are coupled to each-other, as changes in one of the responsibilities may result in additional changes in order for the other responsibilities to be handled properly by that class.

A responsibility can be defined as a reason for change. Whenever we think that some part of our code is potentially a responsibility, we should consider separating it from the class. Let’s say we are working on a project that helps people become more active in their community, and the system needs to have social media integration. It would be a good idea to separate the social media integration responsibility from the other parts of the system, as we should always be prepared for external changes.

## O — Open-Closed Principle

`You should be able to extend a classes behavior, without modifying it.`

This principle is the foundation for building code that is maintainable and reusable. A class follows the OCP if it fulfills these two criteria:

### Open for extension

This ensures that the class behavior can be extended. As requirements change, we should be able to make a class behave in new and different ways, to meet the needs of the new requirements.

### Closed for modification

The source code of such a class is set in stone, no one is allowed to make changes to the code.

### Achieving OCP

OCP can be achieved through abstractions. For example, if we had a system that works with different shapes as classes, we would probably have classes like Circle, Rectangle, etc. In order for a class that depends on one of these classes to implement OCP, we need to introduce a Shape class. Then, wherever we had Dependency Injection, we would inject a Shape instance instead of an instance of a lower-level class. This would give us the luxury of adding new shapes without having to change the dependent classes' source code.

## L — Liskov Substitution Principle

`Derived classes must be substitutable for their base classes.`

### Case study

Let's say we have a Rectangle class, and we have a class that extend it, Square. Let's also say that Rectangle has two methods, set_width and set_height, which sets the width and height of the rectangle.

The problem is that the behavior for the two methods differs between the Rectangle and the Square classes. The reason for that is that a Square, is a Rectangle with equal height and width. So, the two methods will change the same value, whereas for the Rectangle, they will change the width and height respectively, which are different values from each other.

When we are using abstraction (OCP), we want the methods to behave the same for each derived class, and not differently. In this case, we can clearly see that a Square class should not be extending the Rectangle class, because the behavior of the inherited methods differs.

### The solution

Robert C. Martin suggests "design by contract". This means that each method should have preconditions and postconditions defined. Preconditions must hold true in order for a method to execute, and postconditions must hold true after the execution of a method.

## I — Interface Segregation Principle

`Make fine grained interfaces that are client specific.`

In other words, it is better to have many smaller interfaces, than fewer, fatter interfaces.

For example, let's say we had an interface called Animal, which would have eat, sleep and walk methods. This would mean that we have a monolithic interface called Animal, which would not be the perfect abstraction, because some animals can fly.

Breaking this monolithic interface into smaller interfaced based by role, we would get can_eat, can_sleep and can_walk interfaces. This would then make it possible for a species to eat, sleep and for example fly. At a larger scale, microservices are a very similar case, they are pieces of a system separated by responsibilities, instead of being a great monolith.

By breaking down interfaces, we favor composition instead of inheritance, and decoupling over coupling. We favor composition by separating by roles(responsibilities) and decoupling by not coupling derivative classes with unneeded responsibilities inside a monolith.

## D — Dependency Inversion Principle

`Depend on abstractions, not on concretions.`

- High level modules should not depend upon low level modules. Both should depend upon abstractions.
- Abstractions should not depend upon details. Details should depend upon abstractions.

Let's say we have a system that handles authentication through external services such as Google, GitHub, etc. We would have a class for each service: GoogleAuthenticationService, GitHubAuthenticationService, etc. Now, let's say that some place in our system, we need to authenticate our user. To do that, as mentioned, we have several services available. To be able to make use of all the services, we have two possibilities:

We either write a piece of code that adapts each service to the authentication process, or we define an abstraction of the authentication services. The first possibility is a dirty solution that will potentially introduce technical debt in the future; in case a new authentication service is to be integrated to the system, we will need to change the code, which as a result violates the OCP.

 The second possibility is much cleaner, it allows for future addition of services, and changes can be done to each service without changing the integration logic. By defining a AuthenticationService interface and implementing it in each service, we would then be able to use Dependency Injection in our authentication logic and have our authentication method signature look something like this: authenticate(AuthenticationService authenticationService). Then, we could authenticate by a specific service like this: authenticate(new GoogleAuthenticationService). This helps us generalize the authentication logic without having to integrate each service separately.