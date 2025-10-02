#!/bin/bash

# AWS EKS Cluster Creation Script for Fraud Detection System

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CLUSTER_NAME="fraud-detection-eks"
REGION="us-west-2"
NODE_TYPE="t3.medium"
MIN_NODES=2
MAX_NODES=10
DESIRED_NODES=3

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
    echo "  -n, --name NAME          Cluster name [default: fraud-detection-eks]"
    echo "  -r, --region REGION      AWS region [default: us-west-2]"
    echo "  -t, --node-type TYPE     Node instance type [default: t3.medium]"
    echo "  --min-nodes NUM          Minimum number of nodes [default: 2]"
    echo "  --max-nodes NUM          Maximum number of nodes [default: 10]"
    echo "  --desired-nodes NUM      Desired number of nodes [default: 3]"
    echo "  --spot                   Use spot instances"
    echo "  --delete                 Delete the cluster"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --name my-fraud-detection --region us-east-1"
    echo "  $0 --node-type t3.large --min-nodes 3 --max-nodes 15"
    echo "  $0 --spot --delete"
}

# Parse command line arguments
USE_SPOT=false
DELETE_CLUSTER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -t|--node-type)
            NODE_TYPE="$2"
            shift 2
            ;;
        --min-nodes)
            MIN_NODES="$2"
            shift 2
            ;;
        --max-nodes)
            MAX_NODES="$2"
            shift 2
            ;;
        --desired-nodes)
            DESIRED_NODES="$2"
            shift 2
            ;;
        --spot)
            USE_SPOT=true
            shift
            ;;
        --delete)
            DELETE_CLUSTER=true
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

print_status "AWS EKS Cluster Management"
print_status "Cluster: $CLUSTER_NAME"
print_status "Region: $REGION"
print_status "Node Type: $NODE_TYPE"
print_status "Nodes: $DESIRED_NODES (min: $MIN_NODES, max: $MAX_NODES)"

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not found. Please install AWS CLI."
        exit 1
    fi
    
    if ! command -v eksctl &> /dev/null; then
        print_error "eksctl not found. Please install eksctl."
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    print_success "All prerequisites found"
}

# Function to create IAM role for EKS
create_iam_role() {
    print_status "Creating IAM role for fraud detection..."
    
    local role_name="fraud-detection-role"
    local policy_name="fraud-detection-policy"
    
    # Check if role exists
    if aws iam get-role --role-name "$role_name" &> /dev/null; then
        print_warning "IAM role $role_name already exists"
        return
    fi
    
    # Create trust policy
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):oidc-provider/oidc.eks.$REGION.amazonaws.com/id/$(aws eks describe-cluster --name $CLUSTER_NAME --region $REGION --query cluster.identity.oidc.issuer --output text | cut -d'/' -f5)"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.$REGION.amazonaws.com/id/$(aws eks describe-cluster --name $CLUSTER_NAME --region $REGION --query cluster.identity.oidc.issuer --output text | cut -d'/' -f5):sub": "system:serviceaccount:fraud-detection:fraud-detection-sa",
          "oidc.eks.$REGION.amazonaws.com/id/$(aws eks describe-cluster --name $CLUSTER_NAME --region $REGION --query cluster.identity.oidc.issuer --output text | cut -d'/' -f5):aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOF
    
    # Create IAM policy for S3 access
    cat > policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::fraud-detection-models-*",
        "arn:aws:s3:::fraud-detection-models-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData",
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:ListMetrics"
      ],
      "Resource": "*"
    }
  ]
}
EOF
    
    # Create role and policy
    aws iam create-role --role-name "$role_name" --assume-role-policy-document file://trust-policy.json
    aws iam create-policy --policy-name "$policy_name" --policy-document file://policy.json
    aws iam attach-role-policy --role-name "$role_name" --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$policy_name"
    
    # Clean up
    rm trust-policy.json policy.json
    
    print_success "IAM role created: $role_name"
}

# Function to create EKS cluster
create_cluster() {
    print_status "Creating EKS cluster: $CLUSTER_NAME"
    
    # Check if cluster exists
    if aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" &> /dev/null; then
        print_warning "EKS cluster $CLUSTER_NAME already exists"
        return
    fi
    
    # Create cluster configuration
    local cluster_config="cluster-config.yaml"
    cat > "$cluster_config" << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $REGION
  version: "1.27"

iam:
  withOIDC: true
  serviceAccounts:
  - metadata:
      name: fraud-detection-sa
      namespace: fraud-detection
    roleName: fraud-detection-role
    attachPolicyARNs:
    - arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/fraud-detection-policy

managedNodeGroups:
- name: fraud-detection-nodes
  instanceType: $NODE_TYPE
  desiredCapacity: $DESIRED_NODES
  minSize: $MIN_NODES
  maxSize: $MAX_NODES
  volumeSize: 50
  volumeType: gp3
  ssh:
    allow: true
    publicKeyName: $(aws ec2 describe-key-pairs --query 'KeyPairs[0].KeyName' --output text 2>/dev/null || echo "")
  labels:
    node-type: fraud-detection
  tags:
    Environment: production
    Application: fraud-detection
EOF
    
    # Add spot instance configuration if requested
    if [ "$USE_SPOT" = true ]; then
        sed -i '/volumeType: gp3/a\  spot: true' "$cluster_config"
    fi
    
    # Create cluster
    eksctl create cluster -f "$cluster_config"
    
    # Clean up
    rm "$cluster_config"
    
    print_success "EKS cluster created: $CLUSTER_NAME"
}

# Function to install addons
install_addons() {
    print_status "Installing EKS addons..."
    
    # Install AWS Load Balancer Controller
    print_status "Installing AWS Load Balancer Controller..."
    eksctl utils associate-iam-oidc-provider --region="$REGION" --cluster="$CLUSTER_NAME" --approve
    
    # Install cert-manager for SSL certificates
    print_status "Installing cert-manager..."
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
    
    # Install Prometheus Operator
    print_status "Installing Prometheus Operator..."
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set grafana.adminPassword=admin
    
    print_success "EKS addons installed"
}

# Function to delete cluster
delete_cluster() {
    print_status "Deleting EKS cluster: $CLUSTER_NAME"
    
    if ! aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" &> /dev/null; then
        print_warning "EKS cluster $CLUSTER_NAME does not exist"
        return
    fi
    
    # Delete cluster
    eksctl delete cluster --name "$CLUSTER_NAME" --region "$REGION" --wait
    
    # Delete IAM role
    local role_name="fraud-detection-role"
    if aws iam get-role --role-name "$role_name" &> /dev/null; then
        aws iam detach-role-policy --role-name "$role_name" --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/fraud-detection-policy"
        aws iam delete-role --role-name "$role_name"
        print_success "IAM role deleted: $role_name"
    fi
    
    print_success "EKS cluster deleted: $CLUSTER_NAME"
}

# Function to show cluster info
show_cluster_info() {
    print_status "Cluster Information:"
    echo ""
    
    print_status "Cluster Details:"
    aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" --query 'cluster.{Name:name,Version:version,Status:status,Endpoint:endpoint}' --output table
    echo ""
    
    print_status "Node Groups:"
    aws eks list-nodegroups --cluster-name "$CLUSTER_NAME" --region "$REGION" --output table
    echo ""
    
    print_status "Cluster Endpoint:"
    aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" --query 'cluster.endpoint' --output text
    echo ""
    
    print_status "To connect to the cluster:"
    echo "aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME"
}

# Main execution
main() {
    check_prerequisites
    
    if [ "$DELETE_CLUSTER" = true ]; then
        delete_cluster
        exit 0
    fi
    
    create_cluster
    create_iam_role
    install_addons
    show_cluster_info
    
    print_success "EKS cluster setup completed!"
    print_status "Next steps:"
    print_status "1. Update kubeconfig: aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME"
    print_status "2. Deploy fraud detection system: ./scripts/deploy.sh --provider aws --cluster $CLUSTER_NAME --region $REGION"
}

# Run main function
main


