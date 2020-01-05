# Kubernetes course notes

## Kubernetes

- Kubernetes is used for automating deployment, scaling, and management of containerized applications.

## Pods
- A pod is one or more containers with shared storage / network and a specification for how to run the containers.
- Think of pods as a wrapper for a container, for most cases container == pod.
- Can have helper containers for a microservice in a pod (i.e. log processor).
- Best pratice is to have only one container in one pod.
- In general, users shouldn’t need to create Pods directly. They should almost always use controllers.
- By default, pods are only accesible from within the Kubernetes cluster.
- Each pod has a unique IP address, even pods on the same node.
- Pods are "mortal", i.e. they get created, destroy etc. Therefore, it is hard to keep-track of pods ip-adresses etc. Services are used to solve this problem.

<img src="./images/pods.png" width="500"/>

## Services
- In Kubernetes, a service is an abstraction which defines a set of Pods and a policy by which to access them.
- Services enable a loose coupling between dependent Pods.
- Services are long-lived object in Kubernetes.
- Best pratice is to use different services for different types of pods (e.g. one service for the javascript front-end and one for the MYSQL database).
- Services can be connected to internally (type=ClusterIP) or externally (type=NodePort).
- Services connect pods by selectors.
- We can use any selectors name and multiple of them. If we have multiple then it will be an AND operator, i.e. all of the selectors from the service need to match the labels on the pods.

<img src="./images/service.png" width="500"/>

## Replica-sets
- In a deployment setting, rarely deals with pods directly.
- If we deploy pods ourselves, we are responsible for managing them, i.e. restart on crashes etc.
- Instead if we use relica-sets, it will take care of managing the pods, i.e. making sure there is always 3 pods up.
- Should almost always deploy replica-set instead of pods (but replica-set creates and manages pods).
- Replica-set yaml-files is a combination of pod & replica yaml-file, think of it as extra-configuration to Kubernetes.

<img src="./images/replica_set.png" width="500"/>

## Deployments
- One can think about the deployment entity as a sophisticated replica-set. It gives automatic updated with zero-downtime, roll-back etc.
- Deployment creates replica-sets; think of a deployment as an entity that manages replica-sets.
- If we roll-out a new replica-set, the old replica-set will still be online, but with no replicas (i.e. pods). This allow us to roll back to a previous replica-set if our new version causes problems.
- In most cases it's better to use deployments instead of replica-sets, since the API is very similar but deployments has more functionality.

<img src="./images/deployment.png" width="500"/>

## Service discovery
- Kubernetes manges it's own internal dns-service, i.e. mapping services to ip addresses.
- The internal dns service is called kube-dns.
- This allow us to easily find the ip-address of different services and makes communications between services easier.
- Can use the linux command nslookup "database" to find the ip-address of the database-service.
- Note that is only works if we are in the same namespace as the service we are looking up.
- If we are in a different namespace, then we need to use "database.mynamespace" or the fully qualified domain name "database.mynamespace.svc.cluster.local".

<img src="./images/dns.png" width="500"/>

## Namespaces
- Namespaces is a way to partion your resources into seperate areas.
- If we don't give a namespace, the resouces will go to the "default" namespace.
- The same is true if we try to do some command, e.g. kubectl get all, then it will display the resources in the "default" namespace.

<img src="./images/namespace.png" width="500"/>
