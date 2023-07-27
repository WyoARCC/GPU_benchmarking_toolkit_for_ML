# Example Deployment.yml Files for Nautilus Kubernetes Cluster

This repository contains example `deployment.yml` files for deploying applications on the National Research Platform's Nautilus Kubernetes cluster. These files demonstrate the configuration required to run applications in a Kubernetes environment.

## Prerequisites

Before using these `deployment.yml` files, ensure that you have the following prerequisites in place:

1. **Docker Desktop Instance**: A functional and properly configured Docker Desktop instance must be installed on your local machine. Docker Desktop provides the necessary tools to build, run, and manage Docker containers locally.

2. **kubectl Installation**: Install `kubectl`, the Kubernetes command-line tool, on your local machine. `kubectl` allows you to interact with the Kubernetes cluster and manage applications.

3. **Kubeconfig Configuration**: Ensure that `kubectl` is configured with the appropriate kubeconfig file, granting you access to the Nautilus Kubernetes cluster.

4. **Access to a Kubernetes Cluster**: You need access to the Nautilus Kubernetes cluster to deploy applications. Obtain the necessary credentials and permissions to deploy resources to the cluster.

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/WyoARCC/GPU_benchmarking_toolkit_for_ML/edit/main/GPT2Scripting/NRP_Deployment_Files
cd nautilus-k8s-deployments
```

2. Review the example `deployment.yml` files available in the repository. Each file demonstrates the configuration needed to deploy a specific application or service on the Nautilus Kubernetes cluster.

3. Customize the `deployment.yml` files as per your application's requirements. Modify the container image, resource limits, environment variables, and other settings as necessary.

4. Deploy the application to the Nautilus Kubernetes cluster using `kubectl`. Replace `your-deployment-file.yml` with the desired deployment file:

```bash
kubectl apply -f your-deployment-file.yml
```

5. Monitor the application's deployment status and check for any potential errors using `kubectl`:

```bash
kubectl get pods
kubectl describe pod your-pod-name
```

## Note

These example `deployment.yml` files are provided solely for demonstration purposes. They are not intended for direct use in production environments. Modify and adapt the configurations based on your specific application's requirements and the Nautilus Kubernetes cluster's policies.

Ensure that you have the necessary permissions and approvals before deploying any application to a production or shared Kubernetes cluster. Incorrectly configured deployments can result in service disruptions or resource conflicts.

Please use these files responsibly and follow best practices for managing Kubernetes deployments and resources on the Nautilus cluster.

For more information and support, refer to the official Kubernetes documentation and the Nautilus platform documentation.
