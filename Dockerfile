# Base image
FROM python:3.9-slim

# Working directory
WORKDIR /app

# Install netcat for connectivity checks
RUN apt-get update && apt-get install -y --no-install-recommends netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose port for Kafka
EXPOSE 9092

# Command to run the script
ENTRYPOINT ["./entrypoint.sh"]
