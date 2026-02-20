#!/bin/bash

# Display usage information
show_help() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Clean up the ML Inference Services deployment and associated files"
  echo ""
  echo "Options:"
  echo "  -h, --help       Show this help message and exit"
  echo "  -d, --docker     Clean up Docker containers only (keeps plot and output files)"
  echo "  -f, --files      Clean up plots and output files only (keeps Docker containers running)"
  echo "  -y, --yes        Skip confirmation prompts"
  echo ""
  echo "When run without options, this script will clean everything: Docker containers and files"
}

# Default values
CLEAN_DOCKER=true
CLEAN_FILES=true
SKIP_CONFIRMATION=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -d|--docker)
      CLEAN_DOCKER=true
      CLEAN_FILES=false
      shift
      ;;
    -f|--files)
      CLEAN_DOCKER=false
      CLEAN_FILES=true
      shift
      ;;
    -y|--yes)
      SKIP_CONFIRMATION=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      show_help
      exit 1
      ;;
  esac
done

# Define directories to clean
TELEMETRY_OUTPUTS_DIR="./telemetry_mock/outputs"
TELEMETRY_PLOTS_DIR="./telemetry_mock/plots"

# Function to confirm operation
confirm() {
  if [ "$SKIP_CONFIRMATION" = true ]; then
    return 0
  fi
  
  read -p "$1 [y/N] " response
  case "$response" in
    [yY][eE][sS]|[yY]) 
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# Function to clean directories
clean_files() {
  echo "===== Cleaning File Directories ====="
  
  # Clean telemetry mock outputs directory if it exists
  if [ -d "$TELEMETRY_OUTPUTS_DIR" ]; then
    if confirm "Clean telemetry_mock outputs directory?"; then
      echo "Cleaning $TELEMETRY_OUTPUTS_DIR ..."
      rm -rf "${TELEMETRY_OUTPUTS_DIR:?}"/*
    fi
  fi
  
  # Clean telemetry mock plots directory if it exists
  if [ -d "$TELEMETRY_PLOTS_DIR" ]; then
    if confirm "Clean telemetry_mock plots directory?"; then
      echo "Cleaning $TELEMETRY_PLOTS_DIR ..."
      rm -rf "${TELEMETRY_PLOTS_DIR:?}"/*
    fi
  fi
  
  echo "File cleanup completed"
}

# Function to clean Docker resources
clean_docker() {
  echo "===== Cleaning Docker Resources ====="
  
  # Check if Docker is running
  if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running or you don't have permission to use it"
    exit 1
  fi
  
  # Check if docker compose is installed
  if ! command -v docker > /dev/null 2>&1 || ! docker compose version > /dev/null 2>&1; then
    echo "Error: Docker compose plugin is not installed or not accessible"
    exit 1
  fi
  
  # Stop any existing containers
  if confirm "Stop and remove all Docker containers for this project?"; then
    echo "Stopping any existing containers..."
    docker compose down --volumes --remove-orphans
    
    echo "Performing cleanup of Docker resources..."
    # Get the project name (directory name by default)
    PROJECT_NAME=$(basename "$(pwd)")
    
    # Force cleanup of any containers still connected to the network
    NETWORK_NAME="tc32_default_network"
    echo "Looking for containers still attached to $NETWORK_NAME network..."
    ATTACHED_CONTAINERS=$(docker network inspect $NETWORK_NAME 2>/dev/null | grep -o '"Name": "[^"]*"' | grep -v $NETWORK_NAME | cut -d '"' -f 4 | tr '\n' ' ')
    
    if [ -n "$ATTACHED_CONTAINERS" ]; then
      echo "Found containers still attached to the network: $ATTACHED_CONTAINERS"
      echo "Force stopping and removing attached containers..."
      for CONTAINER in $ATTACHED_CONTAINERS; do
        echo "Stopping container $CONTAINER"
        docker stop $CONTAINER 2>/dev/null || true
        echo "Removing container $CONTAINER"
        docker rm -f $CONTAINER 2>/dev/null || true
      done
    fi
    
    # Remove network
    echo "Removing network $NETWORK_NAME"
    docker network rm $NETWORK_NAME 2>/dev/null || true
    
    # Prune all unused networks
    docker network prune -f
    
    # Prune volumes to ensure a clean state
    docker volume prune -f
  fi
  
  echo "Docker cleanup completed"
}

# Main execution
echo "===== TC32 ML Inference Services - Cleanup ====="

# Show what services would be affected if Docker cleanup is enabled
if [ "$CLEAN_DOCKER" = true ] && command -v docker > /dev/null 2>&1; then
  if docker compose ps --services &>/dev/null; then
    ML_SERVICES=$(docker compose config --services | grep -v kafka | grep -v zookeeper | grep -v telemetry-mock 2>/dev/null)
    SERVICE_COUNT=$(echo "$ML_SERVICES" | wc -w)
    
    echo "This will clean up:"
    if [ $SERVICE_COUNT -gt 0 ]; then
      echo "  • $SERVICE_COUNT ML inference service(s)"
    fi
    echo "  • Infrastructure services (Kafka, Zookeeper)"
    if docker compose config --services | grep -q telemetry-mock; then
      echo "  • Telemetry mock service (if enabled)"
    fi
    echo "  • Docker networks and volumes"
  fi
fi

if [ "$CLEAN_FILES" = true ]; then
  echo "  • Output files and plots"
fi
echo ""

# Execute cleanup based on flags
if [ "$CLEAN_DOCKER" = true ]; then
  clean_docker
fi

if [ "$CLEAN_FILES" = true ]; then
  clean_files
fi

echo "===== Cleanup Complete ====="
echo "To restart the services, use: ./run.sh"