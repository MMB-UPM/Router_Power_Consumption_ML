#!/bin/bash

# Parse command line arguments
MOCK_MODE=false
COMPOSE_FILE="docker-compose.yml"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mock) MOCK_MODE=true ;;
        --model-version) MODEL_VERSION="$2"; shift ;;
        --compose-file) COMPOSE_FILE="$2"; shift ;;
        --help) echo "Usage: $0 [--mock] [--model-version VERSION] [--compose-file FILE]"
                echo ""
                echo "Options:"
                echo "  --mock                    Enable telemetry mock service for testing"
                echo "  --model-version VERSION   Specify model version (default: 1.0.0)"
                echo "  --compose-file FILE       Specify docker compose file (default: docker-compose.yml)"
                echo "  --help                    Show this help message"
                echo ""
                echo "This script will start all ML inference services defined in the specified docker-compose file"
                exit 0 ;;
        *) echo "Unknown parameter: $1"
           echo "Use --help for usage information"
           exit 1 ;;
    esac
    shift
done

# Default model version if not specified
MODEL_VERSION=${MODEL_VERSION:-1.0.0}

# Get all ML services from docker-compose.yml (excluding infrastructure services)
echo "===== Discovering ML Services from $COMPOSE_FILE ====="
ML_SERVICES=$(docker compose -f "$COMPOSE_FILE" config --services | grep -v kafka | grep -v zookeeper | grep -v telemetry-mock)

if [ -z "$ML_SERVICES" ]; then
    echo "Error: No ML services found in $COMPOSE_FILE"
    exit 1
fi

echo "Found ML services:"
for SERVICE in $ML_SERVICES; do
    echo "  â€¢ $SERVICE"
done
echo ""

# Verify model files exist for discovered services
echo "===== Verifying Model Files ====="
ALL_MODELS_EXIST=true

for SERVICE in $ML_SERVICES; do
    # Extract environment variables for this service from docker-compose.yml
    CONFIG_OUTPUT=$(docker compose -f "$COMPOSE_FILE" config)
    
    # Extract the service section and then get the environment variables
    MODEL_TYPE=$(echo "$CONFIG_OUTPUT" | awk "/^  $SERVICE:/,/^  [a-zA-Z]/ {print}" | grep -E "^\s*-\s*MODEL_TYPE=" | sed 's/.*MODEL_TYPE=//' | tr -d '"' | head -1)
    ROUTER_TYPE=$(echo "$CONFIG_OUTPUT" | awk "/^  $SERVICE:/,/^  [a-zA-Z]/ {print}" | grep -E "^\s*-\s*ROUTER_TYPE=" | sed 's/.*ROUTER_TYPE=//' | tr -d '"' | head -1)
    
    # If extraction fails, use hardcoded paths based on service name
    if [ -z "$MODEL_TYPE" ] || [ -z "$ROUTER_TYPE" ]; then
        case "$SERVICE" in
            *ra1*|*linear-regression*)
                ROUTER_TYPE="ra"
                MODEL_TYPE="linear_regression"
                ;;
            *ra2*|*random-forest*)
                ROUTER_TYPE="ra"
                MODEL_TYPE="random_forest"
                ;;
            *rb1*|*polynomial*)
                ROUTER_TYPE="rb"
                MODEL_TYPE="polynomial_regression"
                ;;
            *rb2*|*deep-neural*)
                ROUTER_TYPE="rb"
                MODEL_TYPE="deep_neural_network"
                ;;
            *)
                echo "  âš ï¸  Could not determine model configuration for $SERVICE"
                echo ""
                continue
                ;;
        esac
    fi
    
    MODEL_DIR="./ml_inference/models/${ROUTER_TYPE}/${MODEL_TYPE}"
    
    MODEL_FILE="$MODEL_DIR/model_v${MODEL_VERSION}.pkl"
    METADATA_FILE="$MODEL_DIR/metadata_v${MODEL_VERSION}.yaml"
    
    echo "Checking $SERVICE ($ROUTER_TYPE $MODEL_TYPE)..."
    if [ ! -f "$MODEL_FILE" ]; then
        echo "  âŒ Model file not found: $MODEL_FILE"
        ALL_MODELS_EXIST=false
    else
        echo "  âœ… Model file found: $MODEL_FILE"
    fi
    
    if [ ! -f "$METADATA_FILE" ]; then
        echo "  âŒ Metadata file not found: $METADATA_FILE"
        ALL_MODELS_EXIST=false
    else
        echo "  âœ… Metadata file found: $METADATA_FILE"
    fi
    
    # Check for scaler file (optional for some models)
    SCALER_FILE="$MODEL_DIR/scaler_v${MODEL_VERSION}.pkl"
    if [ -f "$SCALER_FILE" ]; then
        echo "  âœ… Scaler file found: $SCALER_FILE"
    else
        echo "  âš ï¸  Scaler file not found: $SCALER_FILE (may not be required for this model)"
    fi
    echo ""
done

if [ "$ALL_MODELS_EXIST" = false ]; then
    echo "Warning: Some required model files are missing. Services may not start correctly."
    echo "Please ensure all model files are in the correct locations before running inference."
    echo ""
fi

SERVICE_COUNT=$(echo "$ML_SERVICES" | wc -w)
echo "===== Starting All ML Inference Services ====="
echo "Using docker compose file: $COMPOSE_FILE"
echo "Using model version: $MODEL_VERSION"
echo "Starting $SERVICE_COUNT ML inference services discovered from $COMPOSE_FILE"
if [ "$MOCK_MODE" = true ]; then
    echo "Mock mode enabled: Telemetry mock service will be started"
fi
echo ""

# Clean outputs and plots directories before starting services
OUTPUTS_DIR="./telemetry_mock/outputs"
PLOTS_DIR="./telemetry_mock/plots"

if [ -d "$OUTPUTS_DIR" ]; then
  echo "Cleaning $OUTPUTS_DIR ..."
  rm -rf "$OUTPUTS_DIR"/*
fi
if [ -d "$PLOTS_DIR" ]; then
  echo "Cleaning $PLOTS_DIR ..."
  rm -rf "$PLOTS_DIR"/*
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or you don't have permission to use it"
  exit 1
fi

# Check if docker compose is installed (new command without dash)
if ! command -v docker > /dev/null 2>&1 || ! docker compose version > /dev/null 2>&1; then
  echo "Error: Docker compose plugin is not installed or not accessible"
  exit 1
fi

# Stop any existing containers from previous runs and clean up thoroughly
echo "Stopping any existing containers..."
docker compose -f "$COMPOSE_FILE" down --volumes --remove-orphans

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

# Remove any networks related to this project
echo "Removing network $NETWORK_NAME"
docker network rm $NETWORK_NAME 2>/dev/null || true

# Make sure the Docker network exists with the right configuration
echo "Setting up Docker network..."
# Try to remove the network first to ensure clean state
docker network rm $NETWORK_NAME 2>/dev/null || true
# Create the network - this MUST succeed before continuing
if ! docker network create $NETWORK_NAME; then
  echo "Failed to create network $NETWORK_NAME"
  echo "Trying to resolve network issues..."
  # Try more aggressive cleanup and retry
  docker system prune -f --volumes
  if ! docker network create $NETWORK_NAME; then
    echo "Critical error: Still cannot create network $NETWORK_NAME"
    exit 1
  fi
fi
echo "Network $NETWORK_NAME successfully created"

# Continue only if the network was created successfully
if ! docker network inspect $NETWORK_NAME &>/dev/null; then
  echo "Critical error: Network $NETWORK_NAME not found after creation"
  exit 1
fi

# Prune all unused networks
docker network prune -f
# Prune volumes to ensure a clean state
docker volume prune -f

# Build and start the containers
echo "Building and starting Docker containers with model version ${MODEL_VERSION}..."

# Use the appropriate command based on mock mode
if [ "$MOCK_MODE" = true ]; then
  MODEL_VERSION=$MODEL_VERSION docker compose -f "$COMPOSE_FILE" --profile telemetry_mock up -d --build --force-recreate
else
  MODEL_VERSION=$MODEL_VERSION docker compose -f "$COMPOSE_FILE" up -d --build --force-recreate
fi

# Check if containers started successfully
if [ $? -eq 0 ]; then
  echo "===== All ML Inference Services Started Successfully ====="
  
  # Display information about all deployed services
  echo "Deployed ML inference services:"
  echo ""
  
  # Get all running containers from docker compose (excluding infrastructure)
  RUNNING_SERVICES=$(docker compose -f "$COMPOSE_FILE" ps --services | grep -v kafka | grep -v zookeeper | grep -v telemetry-mock)
  
  SERVICE_COUNT=0
  
  # Loop through discovered ML services and check if they're running
  for SERVICE in $ML_SERVICES; do
    if echo "$RUNNING_SERVICES" | grep -q "^${SERVICE}$"; then
      # Extract service information dynamically from environment variables
      MODEL_TYPE=$(docker compose -f "$COMPOSE_FILE" exec $SERVICE sh -c 'echo $MODEL_TYPE' 2>/dev/null | tr -d '\r')
      ROUTER_TYPE=$(docker compose -f "$COMPOSE_FILE" exec $SERVICE sh -c 'echo $ROUTER_TYPE' 2>/dev/null | tr -d '\r')
      INPUT_TOPIC=$(docker compose -f "$COMPOSE_FILE" exec $SERVICE sh -c 'echo $KAFKA_INPUT_TOPIC' 2>/dev/null | tr -d '\r')
      OUTPUT_TOPIC=$(docker compose -f "$COMPOSE_FILE" exec $SERVICE sh -c 'echo $KAFKA_OUTPUT_TOPIC' 2>/dev/null | tr -d '\r')
      
      # Get container name
      CONTAINER_NAME=$(docker compose -f "$COMPOSE_FILE" ps -q $SERVICE | xargs docker inspect -f '{{.Name}}' 2>/dev/null | sed 's/^\///' || echo "Not found")
      
      # Create a friendly service name
      if [ -n "$ROUTER_TYPE" ] && [ -n "$MODEL_TYPE" ]; then
        # Capitalize router type for display
        ROUTER_DISPLAY=$(echo "$ROUTER_TYPE" | sed 's/^./\U&/' | sed 's/rb/RB/')
        MODEL_DISPLAY=$(echo "$MODEL_TYPE" | sed 's/_/ /g' | sed 's/\b\w/\U&/g')
        SERVICE_NAME="$ROUTER_DISPLAY $MODEL_DISPLAY"
      else
        SERVICE_NAME="$SERVICE"
      fi
      
      echo "  â€¢ $SERVICE_NAME"
      echo "    Container: $CONTAINER_NAME"
      if [ -n "$INPUT_TOPIC" ] && [ -n "$OUTPUT_TOPIC" ]; then
        echo "    Kafka Topics: $INPUT_TOPIC / $OUTPUT_TOPIC"
      fi
      echo "    Check logs: ./log.sh $SERVICE"
      echo ""
      
      ((SERVICE_COUNT++))
    else
      echo "  âŒ $SERVICE - Failed to start"
      echo ""
    fi
  done
  
  # Check if mock mode is enabled and display info about the mock service
  if [ "$MOCK_MODE" = true ]; then
    if echo "$RUNNING_SERVICES" | grep -q telemetry-mock || docker compose -f "$COMPOSE_FILE" ps telemetry-mock --quiet &>/dev/null; then
      MOCK_CONTAINER=$(docker compose -f "$COMPOSE_FILE" ps -q telemetry-mock | xargs docker inspect -f '{{.Name}}' 2>/dev/null | sed 's/^\///' || echo "telemetry-mock")
      echo "  â€¢ Telemetry Mock Service"
      echo "    Container: $MOCK_CONTAINER"
      echo "    Sends data to all ML services based on test data"
      echo "    Check logs: ./log.sh telemetry-mock"
      echo ""
    else
      echo "  âŒ Telemetry Mock Service - Failed to start"
      echo ""
    fi
  fi
  
  TOTAL_SERVICES=$(echo "$ML_SERVICES" | wc -w)
  echo "Total of $SERVICE_COUNT/$TOTAL_SERVICES inference service(s) running successfully."
  
  if [ $SERVICE_COUNT -eq $TOTAL_SERVICES ]; then
    echo "ðŸŽ‰ All ML inference services are running correctly!"
  else
    echo "âš ï¸  Some services failed to start. Check the logs for more information."
  fi
  
  echo ""
  echo "To stop all services: docker compose -f $COMPOSE_FILE down"
else
  echo "Error: Failed to start the services"
  exit 1
fi
