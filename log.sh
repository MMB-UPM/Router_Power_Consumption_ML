#!/bin/bash

# Default values
FOLLOW=true
TAIL="100"
TIMESTAMPS=false
SERVICE=""
ALL=false

# Help function
show_help() {
  echo "Usage: $0 [OPTIONS] [SERVICE]"
  echo ""
  echo "Monitor logs from ML inference services"
  echo ""
  echo "Options:"
  echo "  -h, --help        Show this help message and exit"
  echo "  -n, --no-follow   Do not follow logs (default is to follow)"
  echo "  -t, --tail N      Show last N lines (default: 100, use 'all' for all logs)"
  echo "  -T, --timestamps  Show timestamps"
  echo "  -a, --all         Show logs from all services"
  echo ""
  echo "Available services:"
  
  # List all available services dynamically
  if command -v docker > /dev/null 2>&1 && docker compose ps --services &>/dev/null; then
    # Get ML services dynamically (same logic as run.sh)
    ML_SERVICES=$(docker compose config --services | grep -v kafka | grep -v zookeeper | grep -v telemetry-mock)
    
    if [ -n "$ML_SERVICES" ]; then
      echo "  ML Inference Services:"
      for service in $ML_SERVICES; do
        echo "    - $service"
      done
    fi
    
    echo "  Infrastructure Services:"
    echo "    - kafka"
    echo "    - zookeeper"
    
    # Check if telemetry-mock service exists in compose file
    if docker compose config --services | grep -q telemetry-mock; then
      echo "    - telemetry-mock"
    fi
  else
    echo "  Docker compose services not available. Start services with run.sh first."
  fi
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -n|--no-follow)
      FOLLOW=false
      shift
      ;;
    -t|--tail)
      TAIL="$2"
      shift 2
      ;;
    -T|--timestamps)
      TIMESTAMPS=true
      shift
      ;;
    -a|--all)
      ALL=true
      shift
      ;;
    *)
      # If no recognized option and no service specified yet, treat as service name
      if [[ -z "$SERVICE" ]]; then
        SERVICE="$1"
        shift
      else
        echo "Error: Unknown option or multiple services specified: $1"
        echo "Use --help for usage information"
        exit 1
      fi
      ;;
  esac
done

# Check if Docker is available
if ! command -v docker &> /dev/null; then
  echo "Error: Docker is not installed or not in PATH"
  exit 1
fi

# Check if Docker Compose services are running
if ! docker compose ps &>/dev/null; then
  echo "Error: Docker Compose services aren't running or the command failed"
  echo "Start the services using ./run.sh first"
  exit 1
fi

# Build the log command
LOG_CMD="docker compose logs"

# Add options
if [ "$FOLLOW" = true ]; then
  LOG_CMD+=" --follow"
fi

if [ "$TIMESTAMPS" = true ]; then
  LOG_CMD+=" --timestamps"
fi

if [ "$TAIL" != "all" ]; then
  LOG_CMD+=" --tail=$TAIL"
fi

# Add service name if specified
if [ "$ALL" = false ] && [ -n "$SERVICE" ]; then
  # Check if the service exists
  if ! docker compose ps "$SERVICE" --quiet &>/dev/null; then
    echo "Error: Service '$SERVICE' not found"
    echo "Available services:"
    docker compose ps --services
    exit 1
  fi
  
  LOG_CMD+=" $SERVICE"
fi

# Display what we're doing
if [ "$ALL" = true ]; then
  echo "Showing logs from all services..."
elif [ -n "$SERVICE" ]; then
  echo "Showing logs from $SERVICE..."
else
  echo "Error: No service specified and --all not used"
  echo "Use -h or --help for usage information"
  exit 1
fi

# Execute the log command
eval "$LOG_CMD"