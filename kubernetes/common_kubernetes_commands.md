# Common Kubernetes commands

## Minikube
- minikube start
    - Starts minikube.
- minikube stop
    - Stops the minikube virtual machine. This may be necessary to do if you have an error when starting.
- minikube delete
    - Do this to completely wipe away the minikube image. Useful if minikube is refusing to start and all else fails.
- minikube env
    - Find out the required environment variables to connect to the docker daemon running in minikube.
- minikube ip
    - Find out the ip address of minikube. Needed for browser access.
- minikube addons list
  - List the available  addons for minikube.
- minikube addons enable \<addon name> 
  - Enables the given addon. Note that we might need to restart minikube.
- minikube dashboard
  - Starts the kubernetes dashboard.

## Kubectl
- kubectl get all
    - List all objects that you’ve created. Pods at first, later, ReplicaSets, Deployments and Services.
- kubectl apply –f \<yaml file>
    - Either creates or updates resources depending on the contents of the yaml file.
- kubectl apply –f .
    - Apply all yaml files found in the current directory.
- kubectl describe pod \<name of pod>
    - Gives full information about the specified pod.
- kubectl exec –it \<pod name> \<command>
    - Execute the specified command in the pod’s container.
- kubectl get (pod | service | replicaset | deployment | hpa)
    - Get all pods, services, replicasets, deployments or horizontal pod autoscaler.
- kubectl get (pv | pvc)
    - Get all persistent-volume or persistent-volume-claim.
- kubectl get po --show-labels
    - Get all pods and their labels.
- kubectl get po --show-labels -l {name}={value}
    - Get all pods matching the specified name:value pair.
- kubectl get \<type> \<name> -o yaml
    - Output the kubernetes type in yaml-format.
- kubectl delete po \<pod name>
    - Delete the named pod. Can also delete svc, rs, deploy.
- kubectl delete po --all
    - Delete all pods (also svc, rs, deploy).
- kubectl logs \<pod name>
    - View the log of the pod (i.e. terminal output from the container). This is very useful for debugging pods.
- kubectl describe node
  - Displays the resource-usage of the node.
- kubectl top (pod | node)
  - Gives memory / CPU usage for the pod / node, i.e. how much is it physically using. Note that we need to enable metrics-server for this command to work.

## Deployment Management
- kubectl rollout status deploy \<name of deployment>
    - Get the status of the named deployment.
- kubectl rollout history deploy \<name of deployment>
    - Get the previous versions of the deployment.
- kubectl rollout undo deploy \<name of deployment>
    - Go back one version in the deployment. Also optionally --to-revision=\<revision-number>. We recommend this is used only in stressful emergency situations! Your YAML will now be out of date with the live deployment! 
