FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the telemetry mock script
COPY telemetry_mock.py .
RUN chmod +x telemetry_mock.py

# Copy data directory
COPY data/ /app/data/

# Create output directories for storing JSON outputs
RUN mkdir -p /app/data/outputs/mock_data /app/data/outputs/test_data

# Set default environment variables
ENV KAFKA_INPUT_TOPICS=ra_1_input,ra_2_input
ENV KAFKA_OUTPUT_TOPICS=ra_1_output,ra_2_output
ENV KAFKA_BROKERS=kafka:9092
ENV INTERVAL=5.0
ENV INPUT_TEMPLATE=/app/data/mock_data/input_ml_metrics.json
ENV LOG_LEVEL=INFO
ENV EXPERIMENT=all

# Command to run the script
CMD ["python", "telemetry_mock.py"]