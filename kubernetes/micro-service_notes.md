# Micro-services

## Momonolitheic approach
- Traditionally, the monolitheic approach to system design is common. Here we have all of the logic of the application in one place.
- Problems:
    - The binary / files get bloated (i.e. big).
    - Hard to make changes without breaking it.
    - Harder to coordinate with other teams for releases etc. 

## Micro-service approach
- Think of it as a extreme version of modularity; building the systems as self-contained compotents.
- Micro-services can function on their own; be developed, deployed etc.
- They can communicate with eachother through well-defined interfaces (e.g. http).
- Micro-services are response for one (if possible) buisness requirements, i.e. single responsibility principle.
- They should be highly cohesive and loosely coupled.
- Highly cohesive: every service should only handle one buisness requirement. Single set of responsible.
- Loosely coupled: where possible, minimize the interface between services. No "spaghetti" communication, i.e. all services talk to each other (unless necessary).

## Databases in micro-services

- The monolith approach uses integration databases, i.e. one big database for many buisness requirements.
- This is a big no-no for micro-services.
- In the microservice architecture configuration, each service operates with its own databases. The databases then synchronize during the operation. 
- This creates a significant challenge in maintaining the consistency of data over numerous microservices (e.g. if two micro-service's database has an user-entity and we add new users).
- We need a flexible approach to keeping track of data transformation to keep data consistent through and through.
- One of the viable solutions can be the Saga pattern. Here’s how it works:
    - Every time service transforms data - an even is published.
    - The other services in the framework take notice and update their databases.

## Pros and Cons of micro-services		

| Pros | Cons |
| --- | --- |
| Greater agility | Needs more collaboration (each team has to cover the whole microservice lifecycle) |
| Faster time to market	| Harder to test and monitor because of the complexity of the architecture |
| Better scalability | Poorer performance, as microservices need to communicate (network latency, message processing, etc.) |
| Faster development cycles (easier deployment and debugging) | Harder to maintain the network (has less fault tolerance, needs more load balancing, etc.)	|	
| Easier to create a CI/CD pipeline for single-responsibility services | Doesn’t work without the proper corporate culture (DevOps culture, automation practices, etc.) |		
| Isolated services have better fault tolerance	| Security issues (harder to maintain transaction safety, distributed communication goes wrong more likely, etc.) |
| Platform- and language agnostic services | |
| Cloud-readiness | |

## Additional comments
- It is an anti-pattern letting the front-end directly communicate with the backend, since the backend might change etc. Better to use a API-gateway; it becomes the single point of entry to the backend.
