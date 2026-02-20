import os
import json
import yaml
import joblib
import logging
import numpy as np
import time
import argparse
from datetime import datetime
from confluent_kafka import Consumer, Producer, KafkaError, KafkaException
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Set up argument parser
parser = argparse.ArgumentParser(description='ML Inference Service')
parser.add_argument('--model-type', type=str, help='Type of ML model to use')
parser.add_argument('--router-type', type=str, help='Type of router')
parser.add_argument('--model-version', type=str, help='Version of the model to use')
parser.add_argument('--kafka-input-topic', type=str, help='Kafka input topic name')
parser.add_argument('--kafka-output-topic', type=str, help='Kafka output topic name')
parser.add_argument('--kafka-brokers', type=str, help='Kafka broker addresses (comma-separated)')
parser.add_argument('--log-level', type=str, help='Logging level', default='INFO')
parser.add_argument('--epsilon', type=float, help='Epsilon value for differentials', default=0.01)
args = parser.parse_args()

# Get configuration from environment variables with CLI args as override
MODEL_TYPE = args.model_type or os.environ.get('MODEL_TYPE')
ROUTER_TYPE = args.router_type or os.environ.get('ROUTER_TYPE')
MODEL_VERSION = args.model_version or os.environ.get('MODEL_VERSION')
INPUT_TOPIC = args.kafka_input_topic or os.environ.get('KAFKA_INPUT_TOPIC')
OUTPUT_TOPIC = args.kafka_output_topic or os.environ.get('KAFKA_OUTPUT_TOPIC')
KAFKA_BROKERS = args.kafka_brokers or os.environ.get('KAFKA_BROKERS', 'kafka:9092')
LOG_LEVEL = args.log_level or os.environ.get('LOG_LEVEL', 'INFO')
EPSILON = float(args.epsilon or os.environ.get('EPSILON', '0.01'))

# Validate required arguments
if not MODEL_TYPE:
    logging.error("MODEL_TYPE is required but not provided. Set via --model-type argument or MODEL_TYPE environment variable.")
    exit(1)

if not ROUTER_TYPE:
    logging.error("ROUTER_TYPE is required but not provided. Set via --router-type argument or ROUTER_TYPE environment variable.")
    exit(1)

if not INPUT_TOPIC:
    logging.error("KAFKA_INPUT_TOPIC is required but not provided. Set via --kafka-input-topic argument or KAFKA_INPUT_TOPIC environment variable.")
    exit(1)

if not OUTPUT_TOPIC:
    logging.error("KAFKA_OUTPUT_TOPIC is required but not provided. Set via --kafka-output-topic argument or KAFKA_OUTPUT_TOPIC environment variable.")
    exit(1)

# Set Logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                   format="%(asctime)s - %(levelname)s - %(message)s")

# Logging configuration information
logging.info(f"Starting ML inference service with parameters:")
logging.info(f"MODEL_TYPE: {MODEL_TYPE}")
logging.info(f"ROUTER_TYPE: {ROUTER_TYPE}")
logging.info(f"MODEL_VERSION: {MODEL_VERSION}")
logging.info(f"INPUT_TOPIC: {INPUT_TOPIC}")
logging.info(f"OUTPUT_TOPIC: {OUTPUT_TOPIC}")
logging.info(f"KAFKA_BROKERS: {KAFKA_BROKERS}")

# Log the configured topics
logging.info(f"Input Kafka topic: {INPUT_TOPIC}")
logging.info(f"Output Kafka topic: {OUTPUT_TOPIC}")

# Define paths for model and metadata files
MODEL_PATH = f"/app/models/{ROUTER_TYPE}/{MODEL_TYPE}/model_v{MODEL_VERSION}.pkl"
METADATA_PATH = f"/app/models/{ROUTER_TYPE}/{MODEL_TYPE}/metadata_v{MODEL_VERSION}.yaml"
SCALER_PATH = f"/app/models/{ROUTER_TYPE}/{MODEL_TYPE}/scaler_v{MODEL_VERSION}.pkl"

# Log the paths being used
logging.info(f"MODEL_PATH: {MODEL_PATH}")
logging.info(f"METADATA_PATH: {METADATA_PATH}")
logging.info(f"SCALER_PATH: {SCALER_PATH}")

# Initialize flags and data structures
NORMALIZE = False
NORMALIZE_FEATURES = None
FEATURES = []
SCALER = None
POLYNOMIAL_FEATURES = None
IS_POLYNOMIAL_MODEL = False

# Load the ML Model and associated files
logging.info("Starting ML inference service...")
logging.info("Loading model from %s...", MODEL_PATH)

try:        
    # Check that model path exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")
        
    # Load model
    MODEL = joblib.load(MODEL_PATH)
    logging.info("Model successfully loaded: %s", type(MODEL).__name__)
    
    # Load metadata file - required for operation
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at path: {METADATA_PATH}")
        
    with open(METADATA_PATH, 'r') as f:
        METADATA = yaml.safe_load(f)
    
    logging.info("Model metadata loaded from %s", METADATA_PATH)
    
    # Extract model information from metadata
    FEATURES = METADATA.get('features', {}).get('input', [])
    
    # Log ALL metadata information
    logging.info("=" * 40)
    logging.info("METADATA SUMMARY")
    logging.info("=" * 40)
    
    # Router information
    router_info = METADATA.get('router', {})
    logging.info("Router: %s", router_info.get('name', 'unknown'))
    logging.info("Sampling interval: %s seconds", router_info.get('sampling_interval_seconds', 'unknown'))
    
    # Model information
    model_info = METADATA.get('model', {})
    stored_model_type = model_info.get('type', 'unknown')
    logging.info("Model type (stored): %s", stored_model_type)
    logging.info("Model type (unified): %s", MODEL_TYPE)
    logging.info("Model version: %s", model_info.get('version', 'unknown'))
    logging.info("Created at: %s", model_info.get('created_at', 'unknown'))
    logging.info("Temporal window: %s", model_info.get('temporal_window', False))
    logging.info("Window size: %s", model_info.get('temporal_window_size', 'N/A'))
    
    # Log hyperparameters if available
    hyperparams = model_info.get('hyperparameters', {})
    if hyperparams:
        logging.info("Model hyperparameters:")
        for param, value in hyperparams.items():
            # Handle nested structures like lists in hidden_layer_sizes
            if isinstance(value, list):
                logging.info("  %s: %s", param, value)
            else:
                logging.info("  %s: %s", param, value)
    else:
        logging.info("No hyperparameters available")
    
    # Features information
    features_info = METADATA.get('features', {})
    logging.info("Input features: %s", features_info.get('input', []))
    logging.info("Target variable: %s", features_info.get('output', 'unknown'))
    
    # Preprocessing information
    preprocessing_info = METADATA.get('preprocessing', {})
    NORM_APPLIED = preprocessing_info.get('normalization_applied', False)
    NORMALIZE_FEATURES = preprocessing_info.get('normalized_features', [])
    logging.info("Normalization applied: %s", NORM_APPLIED)
    logging.info("Normalized features: %s", NORMALIZE_FEATURES)
    
    # Scaler information
    SCALER_INFO = preprocessing_info.get('scaler', {})
    if SCALER_INFO:
        logging.info("Scaler type: %s", SCALER_INFO.get('type', 'unknown'))
        
        # Log scaler parameters
        scaler_params = SCALER_INFO.get('params', {})
        if scaler_params:
            logging.info("Scaler parameters:")
            for param, value in scaler_params.items():
                logging.info("  %s: %s", param, value)
        
        # Log scaler mean values (for StandardScaler)
        scaler_mean = SCALER_INFO.get('mean', [])
        if scaler_mean:
            logging.info("Scaler mean values: %s", scaler_mean)
        
        # Log scaler scale values (for StandardScaler)
        scaler_scale = SCALER_INFO.get('scale', [])
        if scaler_scale:
            logging.info("Scaler scale values: %s", scaler_scale)
    else:
        logging.info("No scaler information available")
    
    # Training data information
    data_info = METADATA.get('data', {})
    logging.info("Training CSV count: %s", data_info.get('training_csv_count', 'unknown'))
    logging.info("Test CSV count: %s", data_info.get('test_csv_count', 'unknown'))
    
    # CSV dates information
    csv_dates = data_info.get('csv_dates', {})
    if csv_dates:
        train_files = csv_dates.get('train_files', {})
        test_files = csv_dates.get('test_files', {})
        
        if train_files:
            # Log training files dates
            logging.info("Training files dates:")
            for file, date in train_files.items():
                # Handle both single dates and lists of dates
                if isinstance(date, list):
                    date_str = ", ".join(str(d) for d in date)
                else:
                    date_str = str(date)
                logging.info("  %s: %s", file, date_str)
        else:
            logging.info("No training files information available")
            
        if test_files:
            # Log testing files dates
            logging.info("Testing files dates:")
            for file, date in test_files.items():
                # Handle both single dates and lists of dates
                if isinstance(date, list):
                    date_str = ", ".join(str(d) for d in date)
                else:
                    date_str = str(date)
                logging.info("  %s: %s", file, date_str)
        else:
            logging.info("No testing files information available")
    else:
        logging.info("No CSV dates information available")
    
    # Performance metrics information
    perf_metrics = METADATA.get('testing_performance_metrics', {})
    if perf_metrics:
        logging.info("-" * 40)
        logging.info("PERFORMANCE METRICS")
        logging.info("-" * 40)
        
        # Helper function to format metric values safely
        def format_metric(value, format_str):
            if value == 'N/A' or value is None:
                return 'N/A'
            try:
                return format_str % value
            except (TypeError, ValueError):
                return str(value)
        
        mse_val = perf_metrics.get('mse', 'N/A')
        mae_val = perf_metrics.get('mae', 'N/A')
        r2_val = perf_metrics.get('r2', 'N/A')
        mape_val = perf_metrics.get('mape', 'N/A')
        smape_val = perf_metrics.get('smape', 'N/A')
        
        logging.info("Mean Squared Error (MSE): %s", format_metric(mse_val, "%.6f"))
        logging.info("Mean Absolute Error (MAE): %s", format_metric(mae_val, "%.6f"))
        logging.info("R² Score: %s", format_metric(r2_val, "%.6f"))
        logging.info("Mean Absolute Percentage Error (MAPE): %s", format_metric(mape_val, "%.4f%%"))
        logging.info("Symmetric Mean Absolute Percentage Error (SMAPE): %s", format_metric(smape_val, "%.4f%%"))
        
        # Feature importance
        if 'feature_importance' in perf_metrics:
            feat_importance = perf_metrics['feature_importance']
            # Sort features by importance (descending)
            sorted_features = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)
            
            logging.info("Feature importance:")
            for feature, importance in sorted_features:
                logging.info("  %s: %.6f", feature, importance)
    else:
        logging.info("No performance metrics available")
    
    logging.info("=" * 40)
    
    # Update normalization flag based on metadata
    NORMALIZE = NORM_APPLIED and NORMALIZE_FEATURES
    
    # Load scaler if normalization is applied
    if NORMALIZE:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at path: {SCALER_PATH}")
            
        SCALER = joblib.load(SCALER_PATH)
        logging.info("Loaded scaler from %s", SCALER_PATH)
    
    # Configure polynomial features for polynomial regression models
    if MODEL_TYPE == 'polynomial_regression':
        IS_POLYNOMIAL_MODEL = True
        # Extract polynomial degree from hyperparameters, default to 2
        polynomial_degree = 2
        if 'hyperparameters' in model_info and 'degree' in model_info['hyperparameters']:
            polynomial_degree = model_info['hyperparameters']['degree']
        
        POLYNOMIAL_FEATURES = PolynomialFeatures(degree=polynomial_degree)
        logging.info("Configured PolynomialFeatures with degree: %d", polynomial_degree)
        logging.info("Polynomial model detected - features will be transformed from %d to expected polynomial combinations", len(FEATURES))
        
except Exception as e:
    logging.error("Error loading model components: %s", e)
    logging.error("Cannot start inference service due to initialization error.")
    exit(1)

# Validate the model and configuration
if not FEATURES:
    logging.error("No input features specified in metadata. Cannot proceed.")
    exit(1)

if NORMALIZE and SCALER is None:
    logging.error("Normalization is enabled but scaler could not be loaded.")
    exit(1)

if IS_POLYNOMIAL_MODEL and POLYNOMIAL_FEATURES is None:
    logging.error("Polynomial model detected but PolynomialFeatures could not be configured.")
    exit(1)

# Validate model compatibility
def validate_model_compatibility():
    """Validate that the loaded model is compatible with the expected configuration"""
    try:
        # Test with dummy data to ensure model works
        dummy_features = {feature: 1.0 for feature in FEATURES}
        X_test = np.array([dummy_features[feature] for feature in FEATURES]).reshape(1, -1)
        
        # Apply normalization if needed (BEFORE polynomial transformation to match training order)
        if NORMALIZE and SCALER:
            norm_indices = [FEATURES.index(f) for f in NORMALIZE_FEATURES if f in FEATURES]
            if norm_indices:
                X_test_norm = X_test.copy()
                features_to_transform = X_test[:, norm_indices]
                normalized_features = SCALER.transform(features_to_transform)
                X_test_norm[:, norm_indices] = normalized_features
                X_test = X_test_norm
                logging.info("Applied normalization for validation")
        
        # Apply polynomial features transformation if needed (AFTER normalization)
        if IS_POLYNOMIAL_MODEL and POLYNOMIAL_FEATURES:
            logging.info("Applying polynomial features transformation for validation...")
            X_test = POLYNOMIAL_FEATURES.fit_transform(X_test)
            logging.info("Features transformed from %d to %d dimensions", len(FEATURES), X_test.shape[1])
        
        # Test prediction
        test_prediction = MODEL.predict(X_test)[0]
        logging.info("Model validation successful. Test prediction: %.6f", test_prediction)
        
        # Validate expected input/output dimensions
        if IS_POLYNOMIAL_MODEL:
            expected_features = X_test.shape[1]  # Use transformed feature count for polynomial models
            logging.info("Polynomial model - using transformed feature count: %d", expected_features)
        else:
            expected_features = len(FEATURES)
            
        if hasattr(MODEL, 'n_features_in_'):
            model_features = MODEL.n_features_in_
            if model_features != expected_features:
                logging.warning(f"Model expects {model_features} features, but we have {expected_features}")
        
        return True
        
    except Exception as e:
        logging.error(f"Model validation failed: {e}")
        return False

# Run model validation
if not validate_model_compatibility():
    logging.error("Model validation failed. Cannot start inference service.")
    exit(1)

logging.info("Model validation completed successfully.")


def normalize_features(features_dict):
    """
    Normalizes specified features according to the configuration from metadata.
    """
    if not NORMALIZE or not NORMALIZE_FEATURES or not SCALER:
        return features_dict  # Return unchanged if no normalization needed
    
    # Create feature array in the order expected by the model
    feature_array = np.array([features_dict[feature] for feature in FEATURES]).reshape(1, -1)
    
    # Get indices of features to normalize
    norm_indices = [FEATURES.index(feature) for feature in NORMALIZE_FEATURES if feature in FEATURES]
    
    if not norm_indices:
        logging.debug("No features to normalize")
        return features_dict
    
    # Create a copy for normalizing
    normalized_array = feature_array.copy()
    
    # Transform selected features only
    features_to_transform = feature_array[:, norm_indices]
    normalized_features = SCALER.transform(features_to_transform)
    normalized_array[0, norm_indices] = normalized_features[0]
    
    logging.debug("Applied normalization to features: %s", NORMALIZE_FEATURES)
    
    # Convert back to dictionary
    normalized_dict = {feature: normalized_array[0, i] for i, feature in enumerate(FEATURES)}
    
    return normalized_dict


def compute_differentials(features_dict, original_prediction):
    """ Computes the differentials for each feature using small perturbations. """
    logging.debug("Computing differentials for input: %s", features_dict)

    # Convert input data to numpy array
    X_original = np.array([features_dict[feature] for feature in FEATURES]).reshape(1, -1)

    differentials = {}
    for feature in FEATURES:
        X_perturbed = X_original.copy()
        idx = FEATURES.index(feature)

        # Add small perturbation (epsilon)
        X_perturbed[0, idx] += EPSILON
        
        # Apply normalization if needed (BEFORE polynomial transformation)
        if NORMALIZE and NORMALIZE_FEATURES and SCALER:
            # Get indices of features to normalize
            norm_indices = [FEATURES.index(f) for f in NORMALIZE_FEATURES if f in FEATURES]
            if norm_indices:
                X_perturbed_norm = X_perturbed.copy()
                features_to_transform = X_perturbed[:, norm_indices]
                normalized_features = SCALER.transform(features_to_transform)
                X_perturbed_norm[:, norm_indices] = normalized_features
                X_perturbed = X_perturbed_norm
        
        # Apply polynomial features transformation if needed (AFTER normalization)
        if IS_POLYNOMIAL_MODEL and POLYNOMIAL_FEATURES:
            X_perturbed = POLYNOMIAL_FEATURES.transform(X_perturbed)
        
        # Make prediction
        perturbed_prediction = MODEL.predict(X_perturbed)[0]

        # Compute derivative (approximate)
        diff = (perturbed_prediction - original_prediction) / EPSILON
        differentials[feature] = float(diff)
        
        logging.debug("Feature %s: diff = %.6f", feature, diff)

    return differentials


def extract_ml_features_from_message(input_data):
    """
    Extracts ML input features from the message in the format of input_ml_metrics.json
    """
    features_dict = {}
    
    # Print the input data for debugging
    logging.debug("Input data: %s", input_data)
    
    # Check if the input has the expected ML metrics structure
    if "input_ml_metrics" in input_data:
        # Extract values from input_ml_metrics
        for metric in input_data["input_ml_metrics"]:
            feature_name = metric["name"]
            feature_value = metric["value"]
            
            # Map the feature names from the JSON to model feature names if necessary
            if feature_name == "node_network_average_received_packet_length":
                mapped_name = "PacketSize_B"
            elif feature_name == "node_network_router_capacity_occupation":
                mapped_name = "Throughput_Percentage"
            else:
                continue  # Skip if not a recognized feature
            
            if mapped_name in FEATURES:
                # Store the feature value in the dictionary
                features_dict[mapped_name] = feature_value
                logging.info("Extracted feature: %s = %s", mapped_name, feature_value)
    
    return features_dict


def process_message(msg):
    """ Processes an incoming Kafka message, performs inference, and sends results. """
    try:
        receive_timestamp = f"{time.time()}"
        process_start_time = time.time()

        # Parse message
        message_value = msg.value().decode("utf-8")
        input_data = json.loads(message_value)

        # Extract features from message using the new structure
        features_dict = extract_ml_features_from_message(input_data)

        # Validate we have all required features
        missing_features = [f for f in FEATURES if f not in features_dict]
        if missing_features:
            logging.error(f"Missing required features in message: {missing_features}")
            logging.error(f"Available features: {list(features_dict.keys())}")
            logging.error(f"Required features: {FEATURES}")
            return
        
        logging.info("Required features: %s", FEATURES)
        logging.info("Extracted features: %s", features_dict)

        # Apply normalization if needed (follows training order)
        normalized_features = features_dict
        if NORMALIZE:
            # Apply normalization for all models that require it
            normalized_features = normalize_features(features_dict)
            if IS_POLYNOMIAL_MODEL:
                logging.info("Normalized features (before polynomial expansion): %s", normalized_features)
            else:
                logging.info("Normalized features: %s", normalized_features)
        
        # Prepare data and make prediction
        X_test = np.array([normalized_features[feature] for feature in FEATURES]).reshape(1, -1)
        logging.debug("Features array for prediction: %s", X_test)
        
        # Apply polynomial features transformation if needed (AFTER normalization)
        if IS_POLYNOMIAL_MODEL and POLYNOMIAL_FEATURES:
            logging.debug("Applying polynomial features transformation...")
            X_test = POLYNOMIAL_FEATURES.transform(X_test)
            logging.debug("Features transformed from %d to %d dimensions", len(FEATURES), X_test.shape[1])
          
        # Make prediction
        prediction = float(MODEL.predict(X_test)[0])
        logging.info("Raw prediction: %.6f", prediction)

        # Compute differentials using original (non-normalized) features for interpretation
        differentials = compute_differentials(features_dict, prediction)
        logging.debug("Computed differentials: %s", differentials)

        # Record timing
        process_end_time = time.time()
        processing_time = process_end_time - process_start_time
        inference_complete_timestamp = f"{time.time()}"

        # Prepare the output message in the format similar to output_ml_metrics.json
        result = input_data.copy()
        
        # Create output_ml_metrics structure if it doesn't exist
        if "output_ml_metrics" not in result:
            result["output_ml_metrics"] = []
        
        # Add power consumption prediction in watts to output_ml_metrics
        result["output_ml_metrics"] = [
            {
                "name": "node_network_power_consumption",
                "description": "Router power consumption prediction",
                "type": "watts",
                "value": prediction
            },
            {
                "name": "node_network_power_consumption_variation_rate_occupation",
                "description": "Router power consumption variation rate over node occupation",
                "type": "power_consumption_variation_rate",
                "value": differentials.get("Throughput_Percentage", 0.0)
            },
            {
                "name": "node_network_power_consumption_variation_rate_packet_length",
                "description": "Router power consumption variation rate over packet length",
                "type": "power_consumption_variation_rate",
                "value": differentials.get("PacketSize_B", 0.0)
            }
        ]
        
        # Add metadata about the ML processing at the same level as input_ml_metrics and output_ml_metrics
        result["ml_metadata"] = {
            "ml_receive_timestamp": receive_timestamp,
            "ml_processing_time_seconds": processing_time,
            "ml_inference_complete_timestamp": inference_complete_timestamp,
            "model_type": MODEL_TYPE,
            "router_type": ROUTER_TYPE,
            "model_version": MODEL_VERSION,
            "feature_coefficients": differentials,
            "normalization_applied": NORMALIZE
        }

        # Send result to output topic with retry mechanism
        if produce_with_retry(OUTPUT_TOPIC, str(msg.key()), json.dumps(result)):
            logging.info(
                "Processed message from %s to %s: Processing Time: %.4f sec | Prediction: %.4f",
                INPUT_TOPIC, OUTPUT_TOPIC, processing_time, prediction
            )
        else:
            logging.error("Failed to send message to output topic after multiple retries")

    except Exception as e:
        logging.error("Error processing message: %s", e)


# Initialize Kafka connection with retry mechanism
def initialize_kafka_clients(max_retries=10, retry_interval=5):
    """Initialize Kafka consumer and producer with retry logic"""
    global CONSUMER, PRODUCER
    
    # Add a startup delay to ensure Kafka topics are available before subscribing
    STARTUP_DELAY_SECONDS = int(os.environ.get('STARTUP_DELAY_SECONDS', '15'))
    logging.info(f"Delaying startup for {STARTUP_DELAY_SECONDS} seconds to allow Kafka and topics to become available...")
    time.sleep(STARTUP_DELAY_SECONDS)
    
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Attempt {attempt}/{max_retries} to connect to Kafka at {KAFKA_BROKERS}")
            
            # Initialize Consumer
            CONSUMER = Consumer({
                'bootstrap.servers': KAFKA_BROKERS,
                'group.id': f'ml_inference_{ROUTER_TYPE}_{MODEL_TYPE}_{MODEL_VERSION}',
                'auto.offset.reset': 'earliest',
                'session.timeout.ms': 10000,  # Longer session timeout
                'socket.timeout.ms': 30000,   # Longer socket timeout
            })
            
            # Test consumer connection by subscribing
            CONSUMER.subscribe([INPUT_TOPIC])
            
            # Initialize Producer
            PRODUCER = Producer({
                'bootstrap.servers': KAFKA_BROKERS,
                'socket.timeout.ms': 30000,   # Longer socket timeout
                'message.timeout.ms': 60000,  # Longer message timeout
            })
            
            # Test producer connection with a lightweight metadata request
            PRODUCER.list_topics(timeout=10)
            
            logging.info("Successfully connected to Kafka")
            return True
            
        except KafkaException as e:
            logging.warning(f"Failed to connect to Kafka (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                logging.info(f"Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
                # Increase interval for next attempt (exponential backoff)
                retry_interval = min(retry_interval * 1.5, 30)
            else:
                logging.error(f"Maximum retries ({max_retries}) reached. Could not connect to Kafka.")
                return False

# Retry mechanism for producing messages
def produce_with_retry(topic, key, value, max_retries=5, initial_backoff=1):
    """Send message to Kafka with retry logic"""
    retry_count = 0
    backoff = initial_backoff
    
    while retry_count < max_retries:
        try:
            PRODUCER.produce(topic, key=key, value=value)
            PRODUCER.flush(timeout=10)
            return True
        except KafkaException as e:
            retry_count += 1
            if retry_count >= max_retries:
                logging.error(f"Failed to produce message after {max_retries} attempts: {e}")
                return False
            
            logging.warning(f"Error producing message (attempt {retry_count}/{max_retries}): {e}")
            logging.info(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)
            # Increase backoff for next retry (exponential backoff)
            backoff = min(backoff * 2, 30)
    
    return False


# Main loop with connection retry and recovery
logging.info("Starting ML inference service with Kafka connection retry mechanism")

# Initialize Kafka connections
if not initialize_kafka_clients():
    logging.error("Could not establish Kafka connections. Exiting.")
    exit(1)

logging.info("Inference service ready. Listening to Kafka topic: %s", INPUT_TOPIC)

try:
    # Keep track of consecutive errors for reconnection logic
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    while True:
        try:
            msg = CONSUMER.poll(timeout=1.0)
            
            # Reset error count on successful poll
            if msg is not None and not msg.error():
                consecutive_errors = 0
                process_message(msg)
                continue
            
            # Handle poll errors or timeout
            if msg is None:
                # Just a timeout, not an error
                continue
                
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition, not an error
                logging.debug("Reached end of partition")
                continue
                
            # Connection errors that might be temporary
            if msg.error().code() == KafkaError._TRANSPORT:
                consecutive_errors += 1
                logging.warning(f"Kafka connection error: {msg.error()} (consecutive: {consecutive_errors})")
                
                # Attempt reconnection after several consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    logging.warning(f"Too many consecutive errors ({consecutive_errors}). Attempting to reconnect...")
                    CONSUMER.close()
                    
                    # Try to reconnect
                    if initialize_kafka_clients():
                        consecutive_errors = 0
                        logging.info("Successfully reconnected to Kafka")
                    else:
                        logging.error("Failed to reconnect to Kafka. Exiting.")
                        break
                
                time.sleep(1)  # Brief pause before next poll
                continue
                
            # Other Kafka errors
            logging.error(f"Kafka error: {msg.error()}")
            
        except KafkaException as e:
            logging.error(f"Kafka exception: {e}")
            consecutive_errors += 1
            
            if consecutive_errors >= max_consecutive_errors:
                logging.error(f"Too many consecutive errors ({consecutive_errors}). Attempting to reconnect...")
                try:
                    CONSUMER.close()
                except:
                    pass
                    
                if not initialize_kafka_clients():
                    logging.error("Failed to reconnect to Kafka after errors. Exiting.")
                    break
                    
                consecutive_errors = 0
                
            time.sleep(1)  # Brief pause before next poll
            
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            time.sleep(1)  # Brief pause before next poll
            
except KeyboardInterrupt:
    logging.info("Shutting down gracefully...")
finally:
    try:
        CONSUMER.close()
        logging.info("Kafka consumer closed.")
    except Exception as e:
        logging.error(f"Error closing Kafka consumer: {e}")