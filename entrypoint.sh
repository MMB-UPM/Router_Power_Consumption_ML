#!/bin/bash

echo "Starting ML inference service with:"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "ROUTER_TYPE: $ROUTER_TYPE"
echo "MODEL_VERSION: $MODEL_VERSION"
echo "KAFKA_INPUT_TOPIC: $KAFKA_INPUT_TOPIC"
echo "KAFKA_OUTPUT_TOPIC: $KAFKA_OUTPUT_TOPIC"
echo "KAFKA_BROKERS: $KAFKA_BROKERS"
echo "LOG_LEVEL: $LOG_LEVEL"

# Define expected model paths based on environment variables
MODEL_DIR="/app/models/$ROUTER_TYPE"/"$MODEL_TYPE"
MODEL_FILE="model_v${MODEL_VERSION}.pkl"
METADATA_FILE="metadata_v${MODEL_VERSION}.yaml"
SCALER_FILE="scaler_v${MODEL_VERSION}.pkl"

# Check if model files exist
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo "Error: Model file not found at $MODEL_DIR/$MODEL_FILE"
  exit 1
fi

if [ ! -f "$MODEL_DIR/$METADATA_FILE" ]; then
  echo "Error: Metadata file not found at $MODEL_DIR/$METADATA_FILE"
  exit 1
fi

# Wait for Kafka to be ready (extra safety measure)
echo "Waiting for Kafka to be ready at $KAFKA_BROKERS..."
RETRY_COUNT=0
MAX_RETRIES=30
RETRY_INTERVAL=2

# Extract host and port from KAFKA_BROKERS
KAFKA_HOST=$(echo ${KAFKA_BROKERS%%,*} | cut -d ':' -f1)
KAFKA_PORT=$(echo ${KAFKA_BROKERS%%,*} | cut -d ':' -f2)

until $(nc -z $KAFKA_HOST $KAFKA_PORT) || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
  echo "Attempt $((RETRY_COUNT+1))/$MAX_RETRIES - Kafka not yet available, waiting ${RETRY_INTERVAL}s..."
  sleep $RETRY_INTERVAL
  RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "Warning: Maximum retries reached waiting for Kafka. Will proceed anyway as the application has its own retry logic."
else
  echo "Kafka appears to be accessible now."
fi

# Run the inference script, passing environment variables
exec python inference.py \
  --model-type "$MODEL_TYPE" \
  --router-type "$ROUTER_TYPE" \
  --model-version "$MODEL_VERSION" \
  --kafka-input-topic "$KAFKA_INPUT_TOPIC" \
  --kafka-output-topic "$KAFKA_OUTPUT_TOPIC" \
  --kafka-brokers "$KAFKA_BROKERS" \
  --log-level "$LOG_LEVEL" \
  --epsilon "$EPSILON"
