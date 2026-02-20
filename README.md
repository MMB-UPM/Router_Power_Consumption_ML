# ML Models for Smart Energy-aware Zero-touch Traffic Engineering

This repository contains code for predicting power consumption in network routers. The solution uses machine learning models to estimate power consumption variation rates based on network metrics. The objective is to provide a way to estimate these values in Network Digital Twin environments, where the goal is to create a digital representation of the network and its components to optimize traffic routing in the real network based on predicted power consumption. This work was conceptualized and developed within the TC3.2 use case of the ACROSS European research project.

## ML Inference Engine Directory Structure

The project contains all necessary components for running the ML inference engine and testing it with mock telemetry data. The directory structure is as follows:

```
/
├── docker-compose.yml           # Container orchestration configuration
├── docker-compose-polynomial.yml # Alternative compose for polynomial models
├── Dockerfile                   # Container definition for inference service
├── entrypoint.sh                # Container startup script
├── log.sh                       # Utility for viewing logs
├── clean.sh                     # Script to clean deployment and files
├── run.sh                       # Script to run the service
├── validate_ml_models.sh        # Script to validate ML model files
├── calculate_metrics.py         # Script to calculate model performance metrics
├── requirements.txt             # Dependencies for deployment
├── README.md                    # This documentation file
├── ml_inference/
│   ├── inference.py             # Main inference service code
│   ├── models/
│   │   ├── ra/                  # RA router-specific models directory
│   │   │   ├── deep_neural_network/
│   │   │   │   ├── model_v1.0.0.pkl
│   │   │   │   ├── metadata_v1.0.0.yaml
│   │   │   │   └── scaler_v1.0.0.pkl
│   │   │   ├── linear_regression/
│   │   │   │   ├── model_v1.0.0.pkl
│   │   │   │   └── metadata_v1.0.0.yaml
│   │   │   ├── polynomial_regression/
│   │   │   │   ├── model_v1.0.0.pkl
│   │   │   │   └── metadata_v1.0.0.yaml
│   │   │   └── random_forest/
│   │   │       ├── model_v1.0.0.pkl
│   │   │       └── metadata_v1.0.0.yaml
│   │   └── rb/                  # RB router-specific models directory
│   │       ├── deep_neural_network/
│   │       │   ├── model_v1.0.0.pkl
│   │       │   ├── metadata_v1.0.0.yaml
│   │       │   └── scaler_v1.0.0.pkl
│   │       ├── linear_regression/
│   │       │   ├── model_v1.0.0.pkl
│   │       │   └── metadata_v1.0.0.yaml
│   │       ├── polynomial_regression/
│   │       │   ├── model_v1.0.0.pkl
│   │       │   └── metadata_v1.0.0.yaml
│   │       └── random_forest/
│   │           ├── model_v1.0.0.pkl
│   │           └── metadata_v1.0.0.yaml
│   └── plots/                   # Pre-generated model validation plots
│       └── ra/linear/
└── telemetry_mock/
    ├── Dockerfile               # Container definition for mock service
    ├── requirements.txt         # Dependencies for mock service
    ├── telemetry_mock.py        # Mock telemetry service implementation
    ├── plots/                   # Directory for validation plots
    ├── outputs/                 # Output data storage
    │   ├── mock_data/           # Outputs from synthetic data
    │   └── test_data/           # Outputs from experiment data
    └── data/                    # Test data for mock service (mounted to /app/data)
        ├── mock_data/           # Template files for synthetic inputs
        │   └── input_ml_metrics.json     # Input message template
        └── test_data/           # Test datasets from experiments
            ├── ra/              # RA router test data
            │   ├── ExperimentoE1_test_data.csv   # Test data from experiment E1
            │   └── ExperimentoE2_test_data.csv   # Test data from experiment E2
            └── rb/              # RB router test data
                ├── ExperimentoE1_test_data.csv   # Test data from experiment E1
                └── ExperimentoE2_test_data.csv   # Test data from experiment E2
```

### Telemetry Mock Service

The telemetry mock service is used for testing the ML inference service without needing actual router telemetry data.

The mock service:
1. Reads test data from the test_data directory
2. Sends it to the ML inference service via Kafka
3. Receives predictions from the ML service
4. Validates results against ground truth values
5. Generates plots in the plots directory

## Core Technologies

- **Python 3.8+**: Main programming language
- **Pandas, NumPy**: Data processing and manipulation
- **Scikit-learn**: Machine learning models (Linear Regression)
- **Docker & Docker Compose**: Containerization and orchestration
- **Kafka**: Message broker for the inference service

## ML Models

The ML Inference service supports multiple machine learning models for power consumption prediction while maintaining a model-agnostic architecture that facilitates easy integration of different model types.

### Available Model Types

The system supports the following model types:

1. **linear_regression** - Linear Regression Model
   - Simple linear regression for baseline predictions
   - Fast inference with minimal computational overhead
   - Good for establishing baseline performance

2. **deep_neural_network** - Deep Neural Network Model
   - Advanced deep learning model with multiple hidden layers
   - Includes feature scaling (scaler_v{VERSION}.pkl required)
   - Optimized for complex pattern recognition

3. **polynomial_regression** - Polynomial Regression Model
   - Polynomial feature expansion for non-linear relationships
   - Captures quadratic and higher-order interactions
   - Balance between complexity and interpretability

4. **random_forest** - Random Forest Model
   - Ensemble method using multiple decision trees
   - Robust to outliers and handles non-linear relationships
   - Good generalization performance
   - Random Forest specifically optimized for RB router telemetry
   - Robust ensemble method with RB-specific hyperparameters
   - Excellent generalization for RB equipment

### Model Features and Input

All models predict router power consumption based on instantaneous network metrics rather than time-series data, utilizing two key features:
- `PacketSize_B`: Average size of received network packets (in bytes)
- `Throughput_Percentage`: Percentage of router capacity currently in use (ranging from 0 to 1)

### Model Organization

Trained on router-specific historical telemetry data, the models deliver real-time power consumption estimations through the inference service. The model files are organized by router type and model type:
- Serialized model: `/models/{ROUTER_TYPE}/{MODEL_TYPE}/model_v{VERSION}.pkl`
- Model metadata: `/models/{ROUTER_TYPE}/{MODEL_TYPE}/metadata_v{VERSION}.yaml`
- Feature scaler: `/models/{ROUTER_TYPE}/{MODEL_TYPE}/scaler_v{VERSION}.pkl` (only for models that require scaling, e.g., deep_neural_network)

The metadata YAML file provides comprehensive model documentation including training datasets, feature specifications, preprocessing parameters, performance metrics, and router compatibility information.

#### RB Router Models

The RB router models are trained on ACROSS energy experiments data from RB equipment and include the following performance metrics:

**Deep Neural Network Model (`deep_neural_network`)**
- **Performance Metrics**:
  - MAE: 1.319
  - MAPE: 1.730%
  - MSE: 3.282
  - R²: 0.836
  - SMAPE: 1.730%
- **Architecture**: 4 hidden layers (64, 32, 16, 8 neurons) with ReLU activation
- **Training**: Adam optimizer, learning rate 0.01, max 1000 iterations
- **Preprocessing**: StandardScaler normalization applied to both features
- **Training Data**: 4 datasets from ExperimentoE1 and ExperimentoE2 (July 2025)
- **Test Data**: 2 datasets from ExperimentoE1 and ExperimentoE2 (June-July 2025)

**Linear Regression Model (`linear_regression`)**
- **Performance Metrics**:
  - MAE: 1.400
  - MAPE: 1.832%
  - MSE: 3.600
  - R²: 0.820
  - SMAPE: 1.831%
- **Configuration**: Standard linear regression with intercept fitting
- **Preprocessing**: No normalization applied
- **Training Data**: Same 4 datasets as deep neural network
- **Test Data**: Same 2 datasets as deep neural network

**Polynomial Regression Model (`polynomial_regression`)**
- **Performance Metrics**:
  - MAE: 1.332
  - MAPE: 1.749%
  - MSE: 3.336
  - R²: 0.833
  - SMAPE: 1.747%
- **Configuration**: Degree 2 polynomial with bias term, no interaction-only terms
- **Preprocessing**: No normalization applied
- **Training Data**: Same 4 datasets as other models
- **Test Data**: Same 2 datasets as other models

**Random Forest Model (`random_forest`)**
- **Performance Metrics**:
  - MAE: 1.302
  - MAPE: 1.709%
  - MSE: 3.223
  - R²: 0.839
  - SMAPE: 1.707%
- **Configuration**: 50 estimators, no max depth limit, random state 42
- **Preprocessing**: No normalization applied
- **Training Data**: Same 4 datasets as other models
- **Test Data**: Same 2 datasets as other models

All RB models were created on July 15, 2025, and use the same input features (`Throughput_Percentage` and `PacketSize_B`) to predict `Energy_Consumption`.

#### RA Router Models

The RA router models are trained on Telefónica's historical router telemetry data and provide established baseline performance for comparison and deployment.

## Deployment Configuration

### Environment Variables

The deployment system uses the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_TYPE` | Type of ML model | linear_regression |
| `ROUTER_TYPE` | Type of router | ra, rb |
| `MODEL_VERSION` | Model version | 1.0.0 |
| `KAFKA_INPUT_TOPIC` | Input Kafka topic name | ra_1_input |
| `KAFKA_OUTPUT_TOPIC` | Output Kafka topic name | ra_1_output |
| `KAFKA_BROKERS` | Kafka broker addresses | kafka:9092 |
| `LOG_LEVEL` | Logging verbosity | INFO |
| `EPSILON` | Value for differential calculations | 0.01 |

### Docker Compose Configuration

The `docker-compose.yml` file defines multiple services:

1. **ML Inference Services**:
   - `ra1-linear-regression`: Linear regression model for ra_1_input/ra_1_output topics
   - `ra2-random-forest`: Random forest model for ra_2_input/ra_2_output topics
   - `rb1-polynomial-regression`: Polynomial regression model for rb_1_input/rb_1_output topics
   - `rb2-deep-neural-network`: Deep neural network model for rb_2_input/rb_2_output topics

2. **Telemetry Mock Service** (optional):
   - Sends test data to ML services
   - Validates responses
   - Enabled with the `--mock` flag or `--profile telemetry_mock` option

3. **Infrastructure**:
   - `kafka`: Message broker (wurstmeister/kafka)
   - `zookeeper`: Required for Kafka operation

## Docker Compose Services

The `docker-compose.yml` file defines the following services:

### 1. ML Inference Services

These are the core ML inference services that run the prediction models:

- **ra1-linear-regression**: RA router with linear regression model
- **ra2-random-forest**: RA router with random forest model  
- **rb1-polynomial-regression**: RB router with polynomial regression model
- **rb2-deep-neural-network**: RB router with deep neural network model

Example configuration for `ra1-linear-regression`:

```yaml
ra1-linear-regression:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: ra1-linear-regression
  environment:
    - MODEL_TYPE=linear_regression
    - ROUTER_TYPE=ra
    - MODEL_VERSION=1.0.0
    - KAFKA_INPUT_TOPIC=ra_1_input
    - KAFKA_OUTPUT_TOPIC=ra_1_output
    - KAFKA_BROKERS=kafka:9092
    - LOG_LEVEL=INFO
  depends_on:
    kafka:
      condition: service_healthy
  restart: unless-stopped
```

**Configuration options**:
- `MODEL_TYPE`: Type of ML model (linear_regression/deep_neural_network/polynomial_regression/random_forest)
- `ROUTER_TYPE`: Router manufacturer/model (ra/rb)
- `MODEL_VERSION`: Version of the model to use (1.0.0)
- `KAFKA_INPUT_TOPIC`: Input Kafka topic name (e.g., ra_1_input, rb_1_input)
- `KAFKA_OUTPUT_TOPIC`: Output Kafka topic name (e.g., ra_1_output, rb_1_output)
- `KAFKA_BROKERS`: Kafka broker connection string
- `LOG_LEVEL`: Logging verbosity

### 2. Telemetry Mock Service

The testing service that simulates router telemetry data:

```yaml
telemetry-mock:
  build:
    context: ./telemetry_mock
    dockerfile: Dockerfile
  container_name: telemetry-mock
  environment:
    - KAFKA_INPUT_TOPICS=ra_1_input,ra_2_input,rb_1_input,rb_2_input
    - KAFKA_OUTPUT_TOPICS=ra_1_output,ra_2_output,rb_1_output,rb_2_output
    - KAFKA_BROKERS=kafka:9092
    - INTERVAL=5.0
    - INPUT_TEMPLATE=/app/data/mock_data/input_ml_metrics.json
    - USE_REAL_DATA=true
    - TEST_DATA_PATH=/app/data/test_data
    - PLOTS_DIR=/app/plots
    - PLOT_INTERVAL=30
  volumes:
    - ./telemetry_mock/data:/app/data
    - ./telemetry_mock/plots:/app/plots
  profiles:
    - telemetry_mock
```

**Configuration options**:
- `KAFKA_INPUT_TOPICS`: Comma-separated list of input topics to send data to
- `KAFKA_OUTPUT_TOPICS`: Comma-separated list of output topics to listen for results
- `INTERVAL`: Time between test messages in seconds (5.0)
- `USE_REAL_DATA`: Whether to use real experiment data (true)
- `TEST_DATA_PATH`: Location of test data files
- `PLOTS_DIR`: Where to save validation plots
- `PLOT_INTERVAL`: How often to generate plots in seconds (30)

### 3. Kafka

Message broker service:

```yaml
kafka:
  image: wurstmeister/kafka
  container_name: kafka
  ports:
    - "9092:9092"
  environment:
    - KAFKA_ADVERTISED_HOST_NAME=kafka
    - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
    - KAFKA_CREATE_TOPICS=ra_1_input:1:1,ra_1_output:1:1,ra_2_input:1:1,ra_2_output:1:1,rb_1_input:1:1,rb_1_output:1:1,rb_2_input:1:1,rb_2_output:1:1
  healthcheck:
    test: ["CMD-SHELL", "kafka-topics.sh --bootstrap-server localhost:9092 --list || exit 1"]
```

**Configuration options**:
- `KAFKA_ADVERTISED_HOST_NAME`: Hostname advertised to clients
- `KAFKA_CREATE_TOPICS`: Topics to create automatically at startup
- `KAFKA_ZOOKEEPER_CONNECT`: Connection string for Zookeeper

### 4. Zookeeper

Required service for Kafka coordination:

```yaml
zookeeper:
  image: wurstmeister/zookeeper
  container_name: zookeeper
  ports:
    - "2181:2181"
  healthcheck:
    test: ["CMD-SHELL", "echo ruok | nc localhost 2181 | grep imok || exit 1"]
```

## Shell Scripts

The deployment directory contains shell scripts to simplify deployment and management:

### run.sh

This script is the main entry point for deploying the ML inference services.

```
Usage: ./run.sh [OPTIONS]

Options:
  --mock                    Enable telemetry mock service for testing
  --model-version VERSION   Specify model version (default: 1.0.0)
  --model-type TYPE         Specify the machine learning model type:
                            linear_regression, deep_neural_network, polynomial_regression, random_forest
  --router-type TYPE        Specify router type: ra/rb (default: ra)
  --help                    Show this help message
```

**Parameters:**
- `--mock`: When provided, the telemetry mock service is also started for testing purposes
- `--model-version VERSION`: Specifies which version of the model to use (e.g., `1.0.0`)
- `--model-type TYPE`: Specifies the machine learning model type to use (linear_regression/deep_neural_network/polynomial_regression/random_forest)
- `--router-type TYPE`: Specifies the router type (ra/rb)
- `--help`: Shows usage information

**Examples:**
```bash
# Start with default settings (linear model, ra router, version 1.0.0)
./run.sh

# Start with model version 2.0.0
./run.sh --model-version 2.0.0

# Start with mock telemetry service for testing
./run.sh --mock

# Start with RB router using deep neural network model
./run.sh --router-type rb --model-type deep_neural_network

# Start with RB router using random forest model
./run.sh --router-type rb --model-type random_forest

# Combine options with mock service for RB testing
./run.sh --mock --model-version 1.0.0 --router-type rb --model-type linear_regression
```

**Behavior:**
1. Checks if Docker and Docker Compose are installed and running
2. Verifies if required model files exist in expected locations
3. Stops any existing containers from previous runs
4. Builds and starts new containers based on provided options
5. Displays information about running services, including container names and Kafka topics

### log.sh

This script provides a convenient way to view and monitor logs from the deployed services.

```
Usage: ./log.sh [OPTIONS] [SERVICE]

Options:
  -h, --help        Show help message and exit
  -n, --no-follow   Do not follow logs (default is to follow)
  -t, --tail N      Show last N lines (default: 100, use 'all' for all logs)
  -T, --timestamps  Show timestamps
  -a, --all         Show logs from all services
```

**Parameters:**
- `SERVICE`: The name of the service to show logs for (e.g., `ra1-linear-regression`, `ra2-random-forest`, `rb1-polynomial-regression`, `rb2-deep-neural-network`, `kafka`, `telemetry-mock`)
- `-h, --help`: Shows usage information and lists available services
- `-n, --no-follow`: Shows logs without continuously following new entries
- `-t, --tail N`: Shows the last N lines of logs (default: 100)
- `-T, --timestamps`: Includes timestamps in the log output
- `-a, --all`: Shows logs from all services

**Examples:**
```bash
# View logs for the ra1-linear-regression service and follow new entries
./log.sh ra1-linear-regression

# View the last 50 lines of logs for kafka
./log.sh -t 50 kafka

# View all logs from all services with timestamps
./log.sh -a -t all -T

# View logs from telemetry-mock without following
./log.sh -n telemetry-mock
```

**Behavior:**
1. Validates that Docker Compose services are running
2. Builds the appropriate Docker Compose log command based on options
3. Executes the log command and displays results

### clean.sh

This script provides a convenient way to clean up the deployment and associated files.

```
Usage: ./clean.sh [OPTIONS]

Options:
  -h, --help       Show this help message and exit
  -d, --docker     Clean up Docker containers only (keeps plot and output files)
  -f, --files      Clean up plots and output files only (keeps Docker containers running)
  -y, --yes        Skip confirmation prompts
```

**Parameters:**
- `-h, --help`: Shows usage information
- `-d, --docker`: Only cleans Docker containers and associated resources (networks, volumes), preserving files
- `-f, --files`: Only cleans telemetry mock plots and outputs directories, preserving Docker containers
- `-y, --yes`: Automatically confirms all operations without prompting

**Examples:**
```bash
# Clean everything (Docker containers and files) with confirmation prompts
./clean.sh

# Clean only Docker containers with confirmation prompts
./clean.sh -d

# Clean only telemetry mock plots and output files with confirmation prompts
./clean.sh -f

# Clean everything without confirmation prompts
./clean.sh --yes

# Clean only files without confirmation prompts
./clean.sh -f --yes
```

**Behavior:**
1. When cleaning files, removes content from telemetry_mock/plots and telemetry_mock/outputs directories
2. When cleaning Docker, stops and removes all project containers, networks, and volumes
3. Presents confirmation prompts for each action unless the --yes flag is provided

## Supported Router Types

Currently, the system supports the following router types:

- **ra**: RA router models (fully implemented)
  - Available models: linear_regression, deep_neural_network, polynomial_regression, random_forest
  - Trained on Telefónica's historical router telemetry data
  - Established baseline performance for production deployment

- **rb**: RB router models (fully implemented)
  - Available models: linear_regression, deep_neural_network, polynomial_regression, random_forest
  - Trained on ACROSS energy experiments data from RB equipment
  - Advanced models with enhanced performance metrics
  - Created July 15, 2025, with comprehensive validation

When specifying the router type through the `--router-type` parameter in `run.sh` or in the Docker Compose configuration, both `ra` and `rb` are supported. The system automatically selects the appropriate model types based on the router type specified.

## Modifying the Deployment

### Adding a New Inference Service

To add a new inference service (e.g., for additional router configurations):

1. **Add a new service definition** to the docker-compose.yml file:

```yaml
rb3-linear-regression:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: rb3-linear-regression
  environment:
    - MODEL_TYPE=linear_regression
    - ROUTER_TYPE=rb
    - MODEL_VERSION=1.0.0
    - KAFKA_INPUT_TOPIC=rb_3_input
    - KAFKA_OUTPUT_TOPIC=rb_3_output
    - KAFKA_BROKERS=kafka:9092
    - LOG_LEVEL=INFO
  depends_on:
    kafka:
      condition: service_healthy
  restart: unless-stopped
```

2. **Update Kafka topic creation** to include the new topics:

```yaml
- KAFKA_CREATE_TOPICS=ra_1_input:1:1,ra_1_output:1:1,ra_2_input:1:1,ra_2_output:1:1,rb_1_input:1:1,rb_1_output:1:1,rb_2_input:1:1,rb_2_output:1:1,rb_3_input:1:1,rb_3_output:1:1
```

3. **Ensure model files** are available in the correct location:
   - For RA: `/models/ra/{MODEL_TYPE}/model_v{VERSION}.pkl`
   - For RB: `/models/rb/{MODEL_TYPE}/model_v{VERSION}.pkl`
   - Metadata: `/models/{ROUTER_TYPE}/{MODEL_TYPE}/metadata_v{VERSION}.yaml`
   - Scaler (if needed): `/models/{ROUTER_TYPE}/{MODEL_TYPE}/scaler_v{VERSION}.pkl`

### Removing a Service

To remove an inference service:

1. Delete or comment out the service definition from docker-compose.yml
2. Update the Kafka topics if necessary

### Modifying Service Configuration

You can configure services in several ways:

1. **Edit docker-compose.yml** to change environment variables directly
2. **Use environment variables** when starting with docker-compose:
   ```bash
   MODEL_VERSION=2.0.0 docker compose up -d ra1-linear-regression
   ```
3. **Use run.sh options**:
   ```bash
   # For RA routers
   ./run.sh --model-version 2.0.0 --router-type ra --model-type linear_regression
   
   # For RB routers
   ./run.sh --model-version 1.0.0 --router-type rb --model-type deep_neural_network
   ```

### Best Practices for Modifications

1. **Always backup** the original docker-compose.yml before making changes
2. **Test changes** with a single service before deploying all services
3. **Ensure model files** exist in the correct locations
4. **Update Kafka topics** when adding new services or changing topic names

## Quick Start Guide

### Prerequisites

- Docker and Docker Compose
- Access to Kafka broker (or use the included development setup)

### Deploying the Inference Service

1. Start the service with Docker Compose:
```bash
./run.sh
```

2. To include the mock telemetry service for testing:
```bash
./run.sh --mock
```

3. View logs for a specific service:
```bash
./log.sh ra1-linear-regression
```

## Inference Service Architecture

The inference service is built around Kafka for message passing:

1. **Input**: Network metrics are received via Kafka on input topics (e.g., `ra_1_input`)
2. **Processing**: The service predicts power consumption based on these metrics
3. **Output**: Results are published to output topics (e.g., `ra_1_output`)

### Input Format

The inference service expects input messages with an `input_ml_metrics` array that contains the features required by the model. Here's an example:

```json
{
  "node_exporter": "r1-service:9100",
  "router_type": "A",
  "epoch_timestamp": "1742376270.1693792",
  "experiment_id": "1",
  "interfaces": ["eth0"],
  "metrics": [...],
  "input_ml_metrics": [
    {
      "name": "node_network_average_received_packet_length",
      "description": "Average length of received packets",
      "type": "length",
      "value": 625000
    },
    {
      "name": "node_network_router_capacity_occupation",
      "description": "Router capacity occupation",
      "type": "percentage",
      "value": 88.43194164673895
    }
  ]
}
```

The service will automatically map the following feature names:
- `node_network_average_received_packet_length` → `PacketSize_B`
- `node_network_router_capacity_occupation` → `Throughput_Percentage`

### Output Format

The service produces output messages that include all original data plus two new sections:

1. `output_ml_metrics`: Contains the power consumption prediction and variation rates
2. `ml_metadata`: Contains processing details and feature differentials

Example output:

```json
{
  "node_exporter": "r1-service:9100",
  "router_type": "A",
  "epoch_timestamp": "1742376270.1693792",
  "experiment_id": "1",
  "interfaces": ["eth0"],
  "metrics": [...],
  "input_ml_metrics": [...],
  "output_ml_metrics": [
    {
      "name": "node_network_power_consumption",
      "description": "Router power consumption prediction",
      "type": "watts",
      "value": 735.7
    },
    {
      "name": "node_network_power_consumption_variation_rate_occupation",
      "description": "Router power consumption variation rate over node occupation",
      "type": "power_consumption_variation_rate",
      "value": 27.87
    },
    {
      "name": "node_network_power_consumption_variation_rate_packet_length",
      "description": "Router power consumption variation rate over packet length",
      "type": "power_consumption_variation_rate",
      "value": 0.0057
    }
  ],
  "ml_metadata": {
    "ml_receive_timestamp": "2025-04-15T10:15:30.123Z",
    "ml_processing_time_seconds": 0.0025,
    "ml_inference_complete_timestamp": "2025-04-15T10:15:30.125Z",
    "feature_coefficients": {
      "Throughput_Percentage": 27.87,
      "PacketSize_B": 0.0057
    }
  }
}
```

Key output elements:
- `node_network_power_consumption`: The predicted power consumption in watts
- `node_network_power_consumption_variation_rate_occupation`: How power consumption changes with traffic load
- `node_network_power_consumption_variation_rate_packet_length`: How power consumption changes with packet size
- `ml_metadata`: Processing information and full feature differentials

## Testing

The repository includes a mock telemetry service that can be used to test the inference service:

1. Start with the telemetry mock profile:
```bash
./run.sh --mock
```

2. The mock service will send test data from the experiments and validate responses.

3. Verification plots will be generated in the `telemetry_mock/plots` directory.
   > **Note**: Plots are only generated when using actual test data files from experiments (CSV files in test_data directory), not when using dummy template files (input_ml_metrics.json and output_ml_metrics.json).

4. The mock service can be configured through environment variables in the docker-compose.yml file:
   - `INTERVAL`: Time between test messages (in seconds)
   - `USE_REAL_DATA`: Whether to use real experiment data
   - `TEST_DATA_PATH`: Path to test data files
   - `PLOT_INTERVAL`: How often to generate verification plots (in seconds)

## Troubleshooting

Common issues:
- **Kafka Connection Issues**: Ensure Kafka and Zookeeper are running correctly
- **Model Loading Errors**: 
  - Verify that model files are correctly placed in the appropriate router directory
  - Model files: `models/{ROUTER_TYPE}/{MODEL_TYPE}/` (linear_regression, deep_neural_network, polynomial_regression, random_forest)
  - Required files: `model_v{VERSION}.pkl`, `metadata_v{VERSION}.yaml`
  - Additional file for neural networks: `scaler_v{VERSION}.pkl`
- **Model Type Compatibility**: Ensure you're using the correct model types
  - Supported types: linear_regression, deep_neural_network, polynomial_regression, random_forest
- **Input Format Issues**: Check that input data matches the expected format for the model

- **Container Startup Failures**: Check logs with `./log.sh <service-name>` for details
