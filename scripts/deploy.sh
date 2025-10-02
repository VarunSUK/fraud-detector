#!/bin/bash

# Fraud Detection System Deployment Script
# Supports AWS EKS, GCP GKE, and Azure AKS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CLUSTER_NAME="fraud-detection-cluster"
REGION="us-west-2"
PROVIDER="aws"
NAMESPACE="fraud-detection"
RELEASE_NAME="fraud-detection"
VALUES_FILE=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --provider PROVIDER    Cloud provider (aws, gcp, azure) [default: aws]"
    echo "  -c, --cluster CLUSTER      Cluster name [default: fraud-detection-cluster]"
    echo "  -r, --region REGION        Region [default: us-west-2]"
    echo "  -n, --namespace NAMESPACE  Kubernetes namespace [default: fraud-detection]"
    echo "  --release-name NAME        Helm release name [default: fraud-detection]"
    echo "  --values FILE              Custom values file"
    echo "  --dry-run                  Perform a dry run"
    echo "  --upgrade                  Upgrade existing deployment"
    echo "  --uninstall                Uninstall the deployment"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --provider aws --cluster my-eks-cluster --region us-east-1"
    echo "  $0 --provider gcp --cluster my-gke-cluster --region us-central1"
    echo "  $0 --provider azure --cluster my-aks-cluster --region eastus"
    echo "  $0 --values custom-values.yaml --upgrade"
}

# Parse command line arguments
DRY_RUN=false
UPGRADE=false
UNINSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--provider)
            PROVIDER="$2"
            shift 2
            ;;
        -c|--cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --release-name)
            RELEASE_NAME="$2"
            shift 2
            ;;
        --values)
            VALUES_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --upgrade)
            UPGRADE=true
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate provider
if [[ ! "$PROVIDER" =~ ^(aws|gcp|azure)$ ]]; then
    print_error "Invalid provider: $PROVIDER. Must be aws, gcp, or azure."
    exit 1
fi

print_status "Fraud Detection System Deployment"
print_status "Provider: $PROVIDER"
print_status "Cluster: $CLUSTER_NAME"
print_status "Region: $REGION"
print_status "Namespace: $NAMESPACE"
print_status "Release: $RELEASE_NAME"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_commands=()
    
    if ! command_exists kubectl; then
        missing_commands+=("kubectl")
    fi
    
    if ! command_exists helm; then
        missing_commands+=("helm")
    fi
    
    case $PROVIDER in
        aws)
            if ! command_exists aws; then
                missing_commands+=("aws")
            fi
            if ! command_exists eksctl; then
                print_warning "eksctl not found. You may need it to create EKS clusters."
            fi
            ;;
        gcp)
            if ! command_exists gcloud; then
                missing_commands+=("gcloud")
            fi
            ;;
        azure)
            if ! command_exists az; then
                missing_commands+=("az")
            fi
            ;;
    esac
    
    if [ ${#missing_commands[@]} -ne 0 ]; then
        print_error "Missing required commands: ${missing_commands[*]}"
        print_error "Please install the missing commands and try again."
        exit 1
    fi
    
    print_success "All prerequisites found"
}

# Function to setup cloud provider authentication
setup_cloud_auth() {
    print_status "Setting up $PROVIDER authentication..."
    
    case $PROVIDER in
        aws)
            print_status "Configuring AWS credentials..."
            aws sts get-caller-identity >/dev/null 2>&1 || {
                print_error "AWS credentials not configured. Please run 'aws configure' or set environment variables."
                exit 1
            }
            
            print_status "Updating kubeconfig for EKS cluster..."
            aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER_NAME" || {
                print_error "Failed to update kubeconfig. Make sure the cluster exists and you have access."
                exit 1
            }
            ;;
        gcp)
            print_status "Configuring GCP credentials..."
            gcloud auth application-default login >/dev/null 2>&1 || {
                print_error "GCP credentials not configured. Please run 'gcloud auth login'."
                exit 1
            }
            
            print_status "Getting GKE cluster credentials..."
            gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$REGION" || {
                print_error "Failed to get cluster credentials. Make sure the cluster exists and you have access."
                exit 1
            }
            ;;
        azure)
            print_status "Configuring Azure credentials..."
            az account show >/dev/null 2>&1 || {
                print_error "Azure credentials not configured. Please run 'az login'."
                exit 1
            }
            
            print_status "Getting AKS cluster credentials..."
            az aks get-credentials --resource-group "${CLUSTER_NAME}-rg" --name "$CLUSTER_NAME" || {
                print_error "Failed to get cluster credentials. Make sure the cluster exists and you have access."
                exit 1
            }
            ;;
    esac
    
    print_success "Cloud authentication configured"
}

# Function to create namespace
create_namespace() {
    print_status "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        print_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        print_success "Namespace $NAMESPACE created"
    fi
}

# Function to add Helm repositories
add_helm_repos() {
    print_status "Adding Helm repositories..."
    
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    print_success "Helm repositories added and updated"
}

# Function to build values file
build_values() {
    local values_file="values-${PROVIDER}.yaml"
    
    if [ -n "$VALUES_FILE" ]; then
        values_file="$VALUES_FILE"
    fi
    
    print_status "Using values file: $values_file"
    
    if [ ! -f "$values_file" ] && [ -z "$VALUES_FILE" ]; then
        print_warning "Values file $values_file not found, using default values"
        values_file="helm/fraud-detection/values.yaml"
    fi
    
    echo "$values_file"
}

# Function to deploy with Helm
deploy_helm() {
    local values_file=$(build_values)
    
    print_status "Deploying with Helm..."
    
    local helm_cmd="helm"
    local helm_args=()
    
    if [ "$DRY_RUN" = true ]; then
        helm_args+=("--dry-run")
        helm_args+=("--debug")
    fi
    
    if [ "$UPGRADE" = true ]; then
        helm_args+=("upgrade")
    else
        helm_args+=("install")
    fi
    
    helm_args+=("$RELEASE_NAME")
    helm_args+=("helm/fraud-detection")
    helm_args+=("--namespace" "$NAMESPACE")
    helm_args+=("--values" "$values_file")
    helm_args+=("--wait")
    helm_args+=("--timeout" "10m")
    
    # Add cloud provider specific values
    helm_args+=("--set" "cloud.provider=$PROVIDER")
    helm_args+=("--set" "cloud.region=$REGION")
    helm_args+=("--set" "cloud.cluster.name=$CLUSTER_NAME")
    
    print_status "Running: $helm_cmd ${helm_args[*]}"
    
    if $helm_cmd "${helm_args[@]}"; then
        print_success "Helm deployment completed successfully"
    else
        print_error "Helm deployment failed"
        exit 1
    fi
}

# Function to uninstall
uninstall_deployment() {
    print_status "Uninstalling fraud detection system..."
    
    if helm uninstall "$RELEASE_NAME" --namespace "$NAMESPACE"; then
        print_success "Uninstall completed successfully"
    else
        print_error "Uninstall failed"
        exit 1
    fi
}

# Function to show deployment status
show_status() {
    print_status "Deployment Status:"
    echo ""
    
    print_status "Pods:"
    kubectl get pods -n "$NAMESPACE" || true
    echo ""
    
    print_status "Services:"
    kubectl get services -n "$NAMESPACE" || true
    echo ""
    
    print_status "Ingresses:"
    kubectl get ingress -n "$NAMESPACE" || true
    echo ""
    
    print_status "Helm releases:"
    helm list -n "$NAMESPACE" || true
}

# Main execution
main() {
    if [ "$UNINSTALL" = true ]; then
        uninstall_deployment
        exit 0
    fi
    
    check_prerequisites
    setup_cloud_auth
    create_namespace
    add_helm_repos
    deploy_helm
    
    if [ "$DRY_RUN" = false ]; then
        print_status "Waiting for deployment to be ready..."
        sleep 30
        show_status
        
        print_success "Deployment completed!"
        print_status "You can check the status with: kubectl get pods -n $NAMESPACE"
        
        if [ "$PROVIDER" = "aws" ]; then
            print_status "To access Grafana, run: kubectl port-forward -n $NAMESPACE svc/fraud-detection-grafana 3000:80"
        fi
    fi
}

# Run main function
main

