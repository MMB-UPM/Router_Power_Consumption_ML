#!/usr/bin/env python3
import argparse
import fnmatch
import glob
import json
import logging
import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, KafkaError, Producer

# Default paths and configurations
DEFAULT_PATHS = {
    "test_data": "/app/data/test_data",
    "template": "/app/data/mock_data/input_ml_metrics.json",
    "plots": "/app/plots",
    "outputs": {
        "mock_data": "/app/outputs/mock_data",
        "test_data": "/app/outputs/test_data"
    }
}

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Telemetry Mock Service')
    parser.add_argument('--kafka-input-topics', type=str, default='', 
                        help='Comma-separated list of Kafka input topic names')
    parser.add_argument('--kafka-output-topics', type=str, default='', 
                        help='Comma-separated list of Kafka output topic names')
    parser.add_argument('--kafka-brokers', type=str, default='kafka:9092', 
                        help='Kafka broker addresses')
    parser.add_argument('--interval', type=float, default=5.0, 
                        help='Interval between messages in seconds')
    parser.add_argument('--input-template', type=str, default=DEFAULT_PATHS["template"], 
                        help='Path to input metrics template JSON')
    parser.add_argument('--test-data', type=str, default=None, 
                        help='Path to test data directory containing .csv files')
    parser.add_argument('--experiment', type=str, default="all", 
                        help='Specific experiment to use (e.g., ExperimentoE1) or "all" for sequential replay')
    parser.add_argument('--use-real-data', action='store_true', 
                        help='Use real test data instead of generated mock data')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        help='Logging level')
    parser.add_argument('--plots-dir', type=str, default=DEFAULT_PATHS["plots"], 
                        help='Directory to save verification plots')
    parser.add_argument('--plot-interval', type=int, default=50, 
                        help='Number of data points to collect before generating a plot')
    parser.add_argument('--version', type=str, default='1.0.0', 
                        help='Model version (for plot titles)')
    return parser.parse_args()

class TelemetryMock:
    """Telemetry Mock Service for simulating router telemetry data and validating model predictions"""
    
    def __init__(self):
        # Parse arguments and set up configuration
        self.args = parse_args()
        self.config = self._load_config()
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        # Print configuration
        self._log_config()
        
        # Initialize data structures
        self._init_data_structures()
        
        # Set up infrastructure (Kafka, directories)
        self._setup_infrastructure()
        
        # Load test data if needed
        if self.config["use_real_data"]:
            self._load_test_data()
    
    def _init_data_structures(self):
        """Initialize internal data structures"""
        # Test data management
        self.test_data_files = []
        self.test_data_records = []
        self.current_file_index = 0
        self.current_record_index = 0
        
        # Router and topic management
        self.router_topic_mapping = {}  # Maps router types to their input/output topics
        
        # Load dummy data template
        self.dummy_data = None
        with open(self.config["input_template"], 'r') as f:
            self.dummy_data = json.load(f)
        
        # Create topic consumers dictionary
        self.topic_consumers = {}
        
        # Initialize router-topic mapping
        self._build_router_topic_mapping()
    
    def _setup_infrastructure(self):
        """Set up required infrastructure (Kafka, directories)"""
        # Create plots directory if it doesn't exist
        os.makedirs(self.config["plots_dir"], exist_ok=True)
        logging.info(f"Created/verified plots directory: {self.config['plots_dir']}")
        
        # Create output directories for saving output data as JSON
        os.makedirs(DEFAULT_PATHS["outputs"]["mock_data"], exist_ok=True)
        os.makedirs(DEFAULT_PATHS["outputs"]["test_data"], exist_ok=True)
        
        # Initialize Kafka producer and consumer
        self.producer = Producer({'bootstrap.servers': self.config["kafka_brokers"]})
        self.consumer = self._init_consumer()
    
    def _load_config(self):
        """Extract configuration from environment variables with CLI args override"""
        # Get the comma-separated list of Kafka input topics
        kafka_input_topics_str = os.environ.get('KAFKA_INPUT_TOPICS', self.args.kafka_input_topics)
        kafka_input_topics = [topic.strip() for topic in kafka_input_topics_str.split(',') if topic.strip()]
        
        # Get the comma-separated list of Kafka output topics
        kafka_output_topics_str = os.environ.get('KAFKA_OUTPUT_TOPICS', self.args.kafka_output_topics)
        kafka_output_topics = [topic.strip() for topic in kafka_output_topics_str.split(',') if topic.strip()]
        
        if not kafka_input_topics:
            logging.warning("No Kafka input topics specified. Telemetry mock service requires at least one input topic.")
        
        if not kafka_output_topics:
            logging.warning("No Kafka output topics specified. Telemetry mock service requires at least one output topic.")
        
        config = {
            "kafka_input_topics": kafka_input_topics,  # List of input topics
            "kafka_output_topics": kafka_output_topics,  # List of output topics
            "kafka_brokers": os.environ.get('KAFKA_BROKERS', self.args.kafka_brokers),
            "interval": float(os.environ.get('INTERVAL', self.args.interval)),
            "input_template": os.environ.get('INPUT_TEMPLATE', self.args.input_template),
            "log_level": os.environ.get('LOG_LEVEL', self.args.log_level),
            "use_real_data": os.environ.get('USE_REAL_DATA', self.args.use_real_data) in [True, 'true', 'True', '1'],
            "test_data_path": os.environ.get('TEST_DATA_PATH', self.args.test_data),
            "experiment": os.environ.get('EXPERIMENT', self.args.experiment),
            "plots_dir": os.environ.get('PLOTS_DIR', self.args.plots_dir),
            "plot_interval": int(os.environ.get('PLOT_INTERVAL', self.args.plot_interval)),
            "version": os.environ.get('VERSION', self.args.version),
        }
        
        return config
        
    def _log_config(self):
        """Log the current configuration"""
        logging.info("Starting Telemetry Mock Service with parameters:")
        for key, value in self.config.items():
            logging.info(f"{key.upper()}: {value}")
        
        if self.config["kafka_input_topics"]:
            logging.info(f"Configured Kafka input topics: {', '.join(self.config['kafka_input_topics'])}")
        else:
            logging.warning("No Kafka input topics configured. Service will not send messages.")
            
        if self.config["kafka_output_topics"]:
            logging.info(f"Configured Kafka output topics: {', '.join(self.config['kafka_output_topics'])}")
        else:
            logging.warning("No Kafka output topics configured. Service will not receive messages.")
    
    def _init_consumer(self):
        """Initialize the Kafka consumer for all output topics"""
        if not self.config["kafka_output_topics"]:
            logging.warning("No Kafka output topics specified. Cannot create consumer.")
            # Return a dummy consumer that does nothing
            return None
        
        consumer = Consumer({
            'bootstrap.servers': self.config["kafka_brokers"],
            'group.id': 'telemetry_mock_consumer',
            'auto.offset.reset': 'latest'
        })
        
        # Subscribe to all output topics
        consumer.subscribe(self.config["kafka_output_topics"])
        
        logging.info(f"Consumer subscribed to topics: {', '.join(self.config['kafka_output_topics'])}")
        
        return consumer
    
    def _load_test_data(self):
        """Load test data files from the specified directory"""
        if not self.config["use_real_data"]:
            logging.info("Using dummy mock data (not loading test data)")
            return
        
        # Determine the test data path
        test_data_path = self.config["test_data_path"] or DEFAULT_PATHS["test_data"]
        
        if not os.path.isdir(test_data_path):
            logging.error(f"Test data directory not found: {test_data_path}")
            return
        
        # Find all .csv files
        csv_files = glob.glob(os.path.join(test_data_path, "**/*.csv"), recursive=True)
        
        if not csv_files:
            logging.warning(f"No .csv files found in {test_data_path}")
            return
        
        # Filter files based on experiment selection
        if self.config["experiment"].lower() != "all":
            csv_files = [f for f in csv_files if self.config["experiment"] in os.path.basename(f)]
            if not csv_files:
                logging.error(f"No .csv files found for experiment {self.config['experiment']}")
                return
        
        # Sort files by router type and experiment order to ensure correct sequence
        def sort_key(file_path):
            filename = os.path.basename(file_path)
            router_path = os.path.dirname(file_path)
            router_type = os.path.basename(router_path).lower()
            
            # Extract experiment ID (E1, E2)
            experiment_part = filename.split('_')[0]  # "ExperimentoE1"
            if experiment_part.startswith("Experimento"):
                exp_id = experiment_part[11:]  # Remove "Experimento" prefix to get "E1"
            else:
                exp_id = experiment_part
            
            # Define experiment order
            exp_order = {'E1': 1, 'E2': 2}
            exp_num = exp_order.get(exp_id, 999)
            
            # Sort by router type first, then by experiment order
            return (router_type, exp_num)
        
        self.test_data_files = sorted(csv_files, key=sort_key)
        logging.info(f"Found {len(self.test_data_files)} test data file(s)")
        for file_path in self.test_data_files:
            logging.info(f"  - {os.path.basename(file_path)}")
            
        # Load the first file
        if not self._load_next_test_file():
            logging.error("Failed to load initial test data file. Using dummy data instead.")
            self.config["use_real_data"] = False
    
    def _load_next_test_file(self):
        """Load the next test data file in the list"""
        total_files = len(self.test_data_files)
        for _ in range(total_files):
            file_path = self.test_data_files[self.current_file_index % total_files]
            self.current_file_path = file_path  # Store current file path for router type detection
            
            try:
                # Load the .csv file
                records = pd.read_csv(file_path, sep=",").to_dict(orient="records")
                
                # Print column names for debugging
                logging.debug(f"Columns in {os.path.basename(file_path)}: {list(records[0].keys())}")
                
                self.test_data_records = records
                logging.info(f"Loaded {len(records)} records from {os.path.basename(file_path)}")
                
                # Update state
                self.current_record_index = 0
                self.current_file_index = (self.current_file_index + 1) % total_files
                self._update_current_experiment(file_path)
                
                return True
                
            except Exception as e:
                logging.warning(f"Failed to load {os.path.basename(file_path)}: {e}")
                self.current_file_index = (self.current_file_index + 1) % total_files
        
        logging.error("Failed to load any test data file")
        return False

    def _update_current_experiment(self, file_path):
        """Extract and update current experiment name from file path"""
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if parts:
            # Extract experiment ID from filename like "ExperimentoE1_test_data.csv" -> "E1"
            experiment_part = parts[0]  # "ExperimentoE1"
            if experiment_part.startswith("Experimento"):
                self.current_experiment = experiment_part[11:]  # Remove "Experimento" prefix to get "E1"
            else:
                self.current_experiment = experiment_part
            logging.info(f"Updated current experiment to: {self.current_experiment}")
    
    def delivery_report(self, err, msg):
        """Callback function for message delivery reports"""
        if err is not None:
            logging.error(f"Message delivery failed: {err}")
        else:
            logging.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    def send_mock_data(self):
        """Send mock data to Kafka input topic"""
        # Get data from appropriate source
        if self.config["use_real_data"]:
            mock_data = self._get_next_test_record()
            if not mock_data:
                logging.error("Failed to get test data record")
                return False
            source_type = "real test data"
        else:
            # Use the dummy template data directly
            mock_data = self.dummy_data.copy()
            mock_data["epoch_timestamp"] = str(time.time())
            source_type = "dummy data"
        
        # Determine router type for this data
        router_type = self._determine_router_type_from_data(mock_data)
        if not router_type:
            logging.error("Could not determine router type for data")
            return False
        
        # Get input topics for this router type
        input_topics, _ = self._get_topics_for_router(router_type)
        if not input_topics:
            logging.warning(f"No input topics configured for router type: {router_type}")
            return False
        
        try:
            # Send to all input topics for this router type
            sent_to_topics = []
            message = json.dumps(mock_data)
            
            for input_topic in input_topics:
                # Send to Kafka
                self.producer.produce(
                    input_topic, 
                    message, 
                    callback=self.delivery_report
                )
                sent_to_topics.append(input_topic)
                
            self.producer.flush()
            
            # Log metrics concisely
            metrics_str = ", ".join([
                f"{m['name']}: {m['value']:.2f}" 
                for m in mock_data.get("input_ml_metrics", [])
            ])
            
            logging.info(f"Sent {source_type} ({router_type}) to topics: {', '.join(sent_to_topics)}: {metrics_str}")
            
            # Log ground truth if available
            if ground_truth := mock_data.get("ground_truth"):
                truth_str = ", ".join([f"{k}: {v:.2f}" for k, v in ground_truth.items()])
                logging.info(f"Ground truth: {truth_str}")
            
            return True
        except Exception as e:
            logging.error(f"Error sending mock data: {e}")
            return False

    def _get_next_test_record(self):
        """Get the next test data record from loaded test data files"""
        if not self.test_data_records:
            logging.debug("No test data records loaded")
            return None
        
        # Check if we need to load next file
        if self.current_record_index >= len(self.test_data_records):
            if not self._load_next_test_file():
                logging.error("Failed to load next test file")
                return None
        
        # Get and format current record
        if self.current_record_index < len(self.test_data_records):
            record = self.test_data_records[self.current_record_index]
            self.current_record_index += 1
            
            # Use the current file path (already stored from _load_next_test_file)
            if hasattr(self, 'current_file_path') and self.current_file_path:
                filename = os.path.basename(self.current_file_path)
                # Extract experiment name (assuming format like "ExperimentoEX_*.csv")
                parts = filename.split('_')
                if parts and len(parts) > 0:
                    experiment_part = parts[0]  # "ExperimentoE1"
                    if experiment_part.startswith("Experimento"):
                        self.current_experiment = experiment_part[11:]  # Remove "Experimento" prefix to get "E1"
                    else:
                        self.current_experiment = experiment_part
            
            # Format test data to match dummy template structure
            mock_data = self.dummy_data.copy()
            mock_data["epoch_timestamp"] = str(time.time())
            
            # Map input metrics
            mock_data["input_ml_metrics"] = [
                {
                    "name": "node_network_average_received_packet_length",
                    "description": "Average length of received packets",
                    "type": "length",
                    "value": float(record["PacketSize_B"])
                },
                {
                    "name": "node_network_router_capacity_occupation",
                    "description": "Router capacity occupation",
                    "type": "percentage",
                    "value": float(record["Throughput_Percentage"])
                },
                {
                    "name": "node_network_average_received_packet_rate",
                    "description": "Average rate of received packets",
                    "type": "rate",
                    "value": float(record["Packet_Rate"])
                },
                {
                    "name": "node_network_router_absolute_capacity_occupation",
                    "description": "Router absolute capacity occupation",
                    "type": "capacity",
                    "value": float(record["Throughput_Gbps"])
                },
            ]
            
            logging.info(f"Current record: {record}")
            
            # Set metrics section
            packet_length = float(record["PacketSize_B"])
            packet_rate = float(record["Packet_Rate"]) if pd.notna(record["Packet_Rate"]) else 0.0
            mock_data["metrics"] = [
                {
                    "name": "node_network_receive_bytes_total",
                    "description": "Network device statistic receive_bytes.",
                    "type": "counter",
                    "values": [
                        {
                            "value": str(int(packet_length * packet_rate)),
                            "labels": [{"name": "device", "value": "eth0"}]
                        }
                    ]
                },
                {
                    "name": "node_network_receive_packets_total", 
                    "description": "Network device statistic receive_packets.",
                    "type": "counter",
                    "values": [
                        {
                            "value": str(int(packet_rate)),
                            "labels": [{"name": "device", "value": "eth0"}]
                        }
                    ]
                }
            ]
            
            # Add ground truth
            if pd.notna(record["Energy_Consumption"]):
                mock_data["ground_truth"] = {
                    "Energy_Consumption": float(record["Energy_Consumption"])
                }
            
            # Add experiment name
            if "experiment_id" in record:
                mock_data["experiment_id"] = record["experiment_id"]
            else:
                # Use the current experiment extracted from filename if not in CSV
                mock_data["experiment_id"] = self.current_experiment
                
            return mock_data
        
        return None
    
    def process_output(self):
        """Read and process messages from the output topic"""
        msg = self.consumer.poll(1.0)
        if not msg or msg.error():
            return

        try:
            output_data = json.loads(msg.value().decode('utf-8'))
            
            if "output_ml_metrics" not in output_data:
                return
            
            # Extract the topic name from the message
            topic = msg.topic()
            # Remove the "_output" suffix to get the base topic name
            base_topic = topic.replace("_output", "")
            
            # Log prediction
            ml_prediction = next((m for m in output_data["output_ml_metrics"] if m["name"] == "node_network_power_consumption"), None)
            
            if ml_prediction:
                logging.info(f"Received prediction from {topic}: {ml_prediction['value']}")
            
            # Log differentials (variation rates)
            for m in output_data.get("output_ml_metrics", []):
                if "variation_rate" in m["name"]:
                    logging.info(f"{m['name']}: {m['value']}")
            
            # Save prediction to file
            self._save_prediction_to_file(output_data, base_topic)
                
        except Exception as e:
            logging.error(f"Error processing output message: {e}")
            
    def _save_prediction_to_file(self, output_data, topic=None):
        """Save prediction output to JSON file"""
        try:
            timestamp_str = datetime.fromtimestamp(float(output_data.get("epoch_timestamp", time.time()))).strftime("%Y%m%d_%H%M%S")
            output_filename = f"prediction_{timestamp_str}.json"
            
            # Use the topic from the message if provided
            if not topic:
                logging.warning("No topic provided for saving prediction, cannot determine output directory")
                return
            
            # Create base output directory based on data source
            base_output_dir = DEFAULT_PATHS["outputs"]["test_data" if self.config["use_real_data"] else "mock_data"]
            
            # Create topic-specific subdirectory
            router_output_dir = os.path.join(base_output_dir, topic)
            os.makedirs(router_output_dir, exist_ok=True)
            
            # Save to topic-specific subdirectory
            with open(os.path.join(router_output_dir, output_filename), 'w') as f:
                json.dump(output_data, f, indent=4)
            logging.info(f"Saved output data to {topic}/{output_filename}")
        except Exception as e:
            logging.error(f"Error saving prediction: {e}")
    
    def plot_outputs_if_enough(self):
        """Generate a plot from outputs if using real test data and enough new outputs are available."""
        if not self.config["use_real_data"]:
            return
        
        # Iterate through all configured router types to generate plots for each
        for router_type in self.router_topic_mapping.keys():
            # Get router display name
            router_display_name = self._get_router_display_name(router_type)
            
            # Get all topics for this router type to determine router IDs
            input_topics, output_topics = self._get_topics_for_router(router_type)
            
            # Process each topic separately to create individual plots
            for topic in output_topics:
                router_id = self._get_router_id_from_topic(topic)
                if router_id is None:
                    continue
                
                # Define path to topic-specific output directory
                base_output_dir = DEFAULT_PATHS["outputs"]["test_data"]
                topic_base = topic.replace("_output", "")
                topic_output_dir = os.path.join(base_output_dir, topic_base)
                
                # Skip if the topic directory doesn't exist
                if not os.path.exists(topic_output_dir):
                    continue
                
                # Create plots directory for this specific router instance
                router_instance_name = f"{router_type}_{router_id}"
                plot_dir = os.path.join(self.config["plots_dir"], router_instance_name)
                os.makedirs(plot_dir, exist_ok=True)
                
                # Get JSON files from topic-specific directory
                json_files = [f for f in os.listdir(topic_output_dir) if f.endswith('.json')]
                if not json_files:
                    continue
                
                # Group data by experiment
                experiments_data = {}
                
                for fname in sorted(json_files):
                    fpath = os.path.join(topic_output_dir, fname)
                    try:
                        with open(fpath, 'r') as f:
                            data = json.load(f)
                        # Extract experiment name directly from the data
                        exp_name = data.get('experiment_id', None)
                        if not exp_name:
                            # Fallback if exp_name is not in the JSON
                            exp_name = 'unknown'
                        
                        gt = None
                        if 'ground_truth' in data and 'Energy_Consumption' in data['ground_truth']:
                            gt = data['ground_truth']['Energy_Consumption']
                        pred = None
                        for m in data.get('output_ml_metrics', []):
                            if m['name'] == 'node_network_power_consumption':
                                pred = m['value']
                                break
                        
                        if gt is not None and pred is not None:
                            if exp_name not in experiments_data:
                                experiments_data[exp_name] = {
                                    'ground_truth': [],
                                    'predictions': []
                                }
                            experiments_data[exp_name]['ground_truth'].append(float(gt))
                            experiments_data[exp_name]['predictions'].append(float(pred))
                            
                    except Exception as e:
                        logging.warning(f"Failed to process {fname}: {e}")
                
                # Generate separate plots for each experiment
                for exp_name, exp_data in experiments_data.items():
                    if not exp_data['ground_truth'] or not exp_data['predictions']:
                        continue
                    
                    all_ground_truth = exp_data['ground_truth']
                    all_predictions = exp_data['predictions']
                    
                    # Generate the plot for this specific experiment
                    plt.figure(figsize=(12, 8))
                    plt.plot(np.arange(len(all_ground_truth)), all_ground_truth, label='Real', alpha=0.7, color='blue', linewidth=2)
                    plt.plot(np.arange(len(all_predictions)), all_predictions, label='Predicted', linestyle='dashed', color='red', linewidth=2)
                    mse = np.mean((np.array(all_ground_truth) - np.array(all_predictions)) ** 2)
                    mae = np.mean(np.abs(np.array(all_ground_truth) - np.array(all_predictions)))
                    
                    # Create plot title with proper router name and ID
                    plot_title = f'Real vs. Predicted Values for Experiment {exp_name}\nRouter: {router_display_name} {router_id}, Version: {self.config["version"]}\nMSE: {mse:.4f}, MAE: {mae:.4f}'
                    plt.title(plot_title, fontsize=25, pad=20)
                    plt.xlabel('Time (s)', fontsize=25)
                    plt.ylabel('Power (W)', fontsize=25)
                    plt.legend(fontsize=25, loc='upper left')
                    plt.xticks(fontsize=20)
                    plt.yticks(fontsize=20)
                    
                    # Set y-axis limits with some margin
                    y_max = max(max(all_ground_truth), max(all_predictions)) * 1.015
                    y_min = min(min(all_ground_truth), min(all_predictions)) * 0.995
                    plt.ylim(y_min, y_max)
                                        
                    # Save the plot in router-instance-specific subdirectory within plots directory (e.g., plots/rb_1, plots/ra_2)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_filename = f"validation_experiment_{exp_name}_{router_type}_{router_id}_{timestamp}.png"
                    plot_path = os.path.join(plot_dir, plot_filename)
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logging.info(f"Generated verification plot for {router_display_name} {router_id} experiment {exp_name}: {plot_filename}")
    
    def _build_router_topic_mapping(self):
        """Build mapping between router types and their corresponding input/output topics"""
        self.router_topic_mapping = {}
        
        # Group topics by router type
        for input_topic in self.config["kafka_input_topics"]:
            # Extract router type (part before first underscore)
            parts = input_topic.split('_')
            if len(parts) >= 2:
                router_type = parts[0].lower()
                
                # Initialize router entry if not exists
                if router_type not in self.router_topic_mapping:
                    self.router_topic_mapping[router_type] = {
                        'input_topics': [],
                        'output_topics': []
                    }
                
                # Add input topic
                self.router_topic_mapping[router_type]['input_topics'].append(input_topic)
        
        # Match output topics to router types
        for output_topic in self.config["kafka_output_topics"]:
            # Extract router type (part before first underscore)
            parts = output_topic.split('_')
            if len(parts) >= 2:
                router_type = parts[0].lower()
                
                # Add output topic if router type exists
                if router_type in self.router_topic_mapping:
                    self.router_topic_mapping[router_type]['output_topics'].append(output_topic)
        
        # Log the mapping for debugging
        logging.info("Router-Topic Mapping:")
        for router_type, topics in self.router_topic_mapping.items():
            logging.info(f"  {router_type.upper()}:")
            logging.info(f"    Input topics: {', '.join(topics['input_topics'])}")
            logging.info(f"    Output topics: {', '.join(topics['output_topics'])}")
    
    def _get_router_display_name(self, router_type):
        """Map router type to display name"""
        router_mapping = {
            'rb': 'RB',
            'ra': 'RA'
        }
        return router_mapping.get(router_type.lower(), router_type.upper())
    
    def _get_router_id_from_topic(self, topic):
        """Extract router ID from topic name (e.g., 'ra_1_input' -> '1')"""
        parts = topic.split('_')
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return None
    
    def _determine_router_type_from_data(self, data_record):
        """
        Determine router type from directory structure.
        
        Expected path structure for real data:
        - /app/data/test_data/{router_type}/file.csv
        
        Expected path structure for dummy data:
        - /app/data/mock_data/input_ml_metrics.json
        """
        
        # For real test data, extract router type from directory structure
        if self.config["use_real_data"] and hasattr(self, 'current_file_path') and self.current_file_path:
            path_parts = os.path.normpath(self.current_file_path).split(os.sep)
            
            # Find test_data in the path and get the next directory
            for i, part in enumerate(path_parts):
                if part == 'test_data':
                    # The router type should be the immediate subdirectory
                    if i + 1 < len(path_parts):
                        router_type = path_parts[i + 1].lower()
                        return router_type
                    break
        
        # For dummy data, always use "mock" as router type
        return "mock"
    
    def _get_topics_for_router(self, router_type):
        """Get input and output topics for a specific router type"""
        if router_type not in self.router_topic_mapping:
            logging.warning(f"Router type '{router_type}' not found in mapping")
            return [], []
        
        mapping = self.router_topic_mapping[router_type]
        return mapping['input_topics'], mapping['output_topics']
    
    def run(self):
        """Main loop for the telemetry mock service"""
        logging.info(f"Telemetry mock service started. Sending data every {self.config['interval']} seconds.")
        try:
            output_count = 0
            
            while True:
                if self.send_mock_data():
                    time.sleep(self.config["interval"])
                self.process_output()
                output_count += 1
                
                # Check plot interval outside the plotting function for clarity
                if (self.config["use_real_data"] and self.config["plot_interval"] > 0 and
                    output_count % self.config["plot_interval"] == 0):
                    logging.info(f"Plot interval reached ({self.config['plot_interval']} outputs). Generating plot...")
                    self.plot_outputs_if_enough()
        except KeyboardInterrupt:
            logging.info("Shutting down gracefully...")
        finally:
            self.consumer.close()
            logging.info("Kafka consumer closed.")

if __name__ == "__main__":
    mock_service = TelemetryMock()
    mock_service.run()