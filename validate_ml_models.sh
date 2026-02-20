#!/bin/bash

# validate_ml_models.sh
# Script to validate ML models structure, metadata, and consistency across routers

set -uo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_MODELS_DIR="${SCRIPT_DIR}/ml_inference/models"
ERRORS=0
WARNINGS=0
ERROR_LOG=()
WARNING_LOG=()

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to log errors
log_error() {
    local message=$1
    ERROR_LOG+=("$message")
    ((ERRORS++))
    print_color "$RED" "ERROR: $message"
}

# Function to log warnings
log_warning() {
    local message=$1
    WARNING_LOG+=("$message")
    ((WARNINGS++))
    print_color "$YELLOW" "WARNING: $message"
}

# Function to log info
log_info() {
    local message=$1
    print_color "$BLUE" "INFO: $message"
}

# Function to log success
log_success() {
    local message=$1
    print_color "$GREEN" "SUCCESS: $message"
}

# Function to check if a file exists
check_file_exists() {
    local filepath=$1
    local description=$2
    
    if [[ ! -f "$filepath" ]]; then
        log_error "$description not found: $filepath"
        return 1
    fi
    return 0
}

# Function to validate YAML syntax
validate_yaml_syntax() {
    local yaml_file=$1
    
    if ! python3 -c "
import yaml
import sys
try:
    with open('$yaml_file', 'r') as f:
        yaml.safe_load(f)
except yaml.YAMLError as e:
    print(f'YAML syntax error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'Error reading file: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_error "Invalid YAML syntax in $yaml_file"
        return 1
    fi
    return 0
}

# Function to extract YAML value using yq or python
get_yaml_value() {
    local yaml_file=$1
    local key_path=$2
    
    python3 -c "
import yaml
import sys
try:
    with open('$yaml_file', 'r') as f:
        data = yaml.safe_load(f)
    
    keys = '$key_path'.split('.')
    value = data
    for key in keys:
        if key in value:
            value = value[key]
        else:
            print('null')
            sys.exit(0)
    
    if value is None:
        print('null')
    else:
        print(value)
except Exception as e:
    print('null')
" 2>/dev/null
}

# Function to pretty print YAML
pretty_print_yaml() {
    local yaml_file=$1
    local router=$2
    local model_type=$3
    
    print_color "$PURPLE" "╔══════════════════════════════════════════════════════════════════════════════╗"
    print_color "$PURPLE" "║ Router: $(printf '%-20s' "$router") │ Model: $(printf '%-35s' "$model_type") ║"
    print_color "$PURPLE" "╚══════════════════════════════════════════════════════════════════════════════╝"
    
    if [[ ! -f "$yaml_file" ]]; then
        print_color "$RED" "Metadata file not found"
        echo
        return
    fi
    
    python3 -c "
import yaml
import sys
from datetime import datetime

def format_list(items, indent=2):
    if not items:
        return '[]'
    if len(items) == 1:
        return f'[{items[0]}]'
    formatted = '[\n'
    for item in items:
        formatted += ' ' * (indent + 2) + f'• {item}\n'
    formatted += ' ' * indent + ']'
    return formatted

def format_dict(d, indent=2):
    if not d:
        return '{}'
    formatted = ''
    for key, value in d.items():
        if isinstance(value, dict):
            formatted += ' ' * indent + f'{key}:\n'
            formatted += format_dict(value, indent + 2) + '\n'
        elif isinstance(value, list):
            formatted += ' ' * indent + f'{key}: {format_list(value, indent)}\n'
        else:
            formatted += ' ' * indent + f'{key}: {value}\n'
    return formatted.rstrip()

try:
    with open('$yaml_file', 'r') as f:
        data = yaml.safe_load(f)
    
    # Model Information
    print('Model Information')
    print('  Type:', data.get('model', {}).get('type', 'N/A'))
    print('  Version:', data.get('model', {}).get('version', 'N/A'))
    print('  Created:', data.get('model', {}).get('created_at', 'N/A'))
    print('  Router:', data.get('router', {}).get('name', 'N/A'))
    print()
    
    # Features
    print('Features')
    input_features = data.get('features', {}).get('input', [])
    output_feature = data.get('features', {}).get('output', 'N/A')
    print(f'  Input Features: {format_list(input_features)}')
    print(f'  Output Feature: {output_feature}')
    print()
    
    # Data Information
    print('Dataset Information')
    data_info = data.get('data', {})
    print(f'  Training CSV Count: {data_info.get(\"training_csv_count\", \"N/A\")}')
    print(f'  Test CSV Count: {data_info.get(\"test_csv_count\", \"N/A\")}')
    print()
    
    # Preprocessing
    print('Preprocessing')
    preprocessing = data.get('preprocessing', {})
    norm_applied = preprocessing.get('normalization_applied', False)
    print(f'  Normalization Applied: {\"Yes\" if norm_applied else \"No\"}')
    
    if norm_applied:
        scaler = preprocessing.get('scaler', {})
        scaler_type = scaler.get('type', 'N/A')
        print(f'  Scaler Type: {scaler_type}')
        if 'mean' in scaler:
            means = scaler['mean']
            if isinstance(means, list) and len(means) <= 3:
                print(f'  Scaler Means: {means}')
        if 'scale' in scaler:
            scales = scaler['scale']
            if isinstance(scales, list) and len(scales) <= 3:
                print(f'  Scaler Scales: {scales}')
    print()
    
    # Model Hyperparameters
    print('Hyperparameters')
    hyperparams = data.get('model', {}).get('hyperparameters', {})
    if hyperparams:
        for key, value in hyperparams.items():
            if isinstance(value, list):
                print(f'  {key}: {format_list(value)}')
            else:
                print(f'  {key}: {value}')
    else:
        print('  No hyperparameters found')
    print()
    
    # Performance Metrics
    print('Performance Metrics')
    metrics = data.get('testing_performance_metrics', {})
    if metrics:
        print(f'  MAE (Mean Absolute Error): {metrics.get(\"mae\", \"N/A\"):.6f}' if isinstance(metrics.get('mae'), (int, float)) else f'  MAE: {metrics.get(\"mae\", \"N/A\")}')
        print(f'  MAPE (Mean Absolute Percentage Error): {metrics.get(\"mape\", \"N/A\"):.6f}%' if isinstance(metrics.get('mape'), (int, float)) else f'  MAPE: {metrics.get(\"mape\", \"N/A\")}')
        print(f'  MSE (Mean Squared Error): {metrics.get(\"mse\", \"N/A\"):.6f}' if isinstance(metrics.get('mse'), (int, float)) else f'  MSE: {metrics.get(\"mse\", \"N/A\")}')
        print(f'  R² (Coefficient of Determination): {metrics.get(\"r2\", \"N/A\"):.6f}' if isinstance(metrics.get('r2'), (int, float)) else f'  R²: {metrics.get(\"r2\", \"N/A\")}')
        print(f'  SMAPE (Symmetric Mean Absolute Percentage Error): {metrics.get(\"smape\", \"N/A\"):.6f}%' if isinstance(metrics.get('smape'), (int, float)) else f'  SMAPE: {metrics.get(\"smape\", \"N/A\")}')
    else:
        print('  No performance metrics found')
    
except Exception as e:
    print(f'Error parsing YAML: {e}')
" 2>/dev/null
    echo
}

# Function to validate required metadata fields
validate_metadata_structure() {
    local yaml_file=$1
    local router=$2
    local model_type=$3
    
    log_info "Validating metadata structure for $router/$model_type"
    
    # Required top-level fields
    local required_fields=(
        "data"
        "features"
        "model"
        "preprocessing"
        "router"
        "testing_performance_metrics"
    )
    
    local missing_fields=()
    
    for field in "${required_fields[@]}"; do
        local value=$(get_yaml_value "$yaml_file" "$field")
        if [[ "$value" == "null" ]]; then
            missing_fields+=("$field")
        fi
    done
    
    if [[ ${#missing_fields[@]} -gt 0 ]]; then
        log_error "$router/$model_type: Missing required top-level fields: ${missing_fields[*]}"
    fi
    
    # Validate specific nested fields
    validate_nested_fields "$yaml_file" "$router" "$model_type"
}

# Function to validate nested metadata fields
validate_nested_fields() {
    local yaml_file=$1
    local router=$2
    local model_type=$3
    
    # Validate data section
    local test_csv_count=$(get_yaml_value "$yaml_file" "data.test_csv_count")
    local training_csv_count=$(get_yaml_value "$yaml_file" "data.training_csv_count")
    
    if [[ "$test_csv_count" == "null" ]]; then
        log_error "$router/$model_type: Missing data.test_csv_count"
    fi
    
    if [[ "$training_csv_count" == "null" ]]; then
        log_error "$router/$model_type: Missing data.training_csv_count"
    fi
    
    # Validate features section
    local input_features=$(get_yaml_value "$yaml_file" "features.input")
    local output_feature=$(get_yaml_value "$yaml_file" "features.output")
    
    if [[ "$input_features" == "null" ]]; then
        log_error "$router/$model_type: Missing features.input"
    fi
    
    if [[ "$output_feature" == "null" ]]; then
        log_error "$router/$model_type: Missing features.output"
    fi
    
    # Validate model section
    local model_version=$(get_yaml_value "$yaml_file" "model.version")
    local model_created_at=$(get_yaml_value "$yaml_file" "model.created_at")
    local model_type_field=$(get_yaml_value "$yaml_file" "model.type")
    
    if [[ "$model_version" == "null" ]]; then
        log_error "$router/$model_type: Missing model.version"
    fi
    
    if [[ "$model_created_at" == "null" ]]; then
        log_error "$router/$model_type: Missing model.created_at"
    fi
    
    if [[ "$model_type_field" == "null" ]]; then
        log_error "$router/$model_type: Missing model.type"
    elif [[ "$model_type_field" != "$model_type" ]]; then
        log_error "$router/$model_type: model.type mismatch - expected '$model_type', found '$model_type_field'"
    fi
    
    # Validate router section
    local router_name=$(get_yaml_value "$yaml_file" "router.name")
    
    if [[ "$router_name" == "null" ]]; then
        log_error "$router/$model_type: Missing router.name"
    elif [[ "$router_name" != "$router" ]]; then
        log_error "$router/$model_type: router.name mismatch - expected '$router', found '$router_name'"
    fi
    
    # Validate preprocessing section and scaler requirements
    validate_scaler_requirements "$yaml_file" "$router" "$model_type"
    
    # Validate performance metrics
    validate_performance_metrics "$yaml_file" "$router" "$model_type"
}

# Function to validate scaler requirements
validate_scaler_requirements() {
    local yaml_file=$1
    local router=$2
    local model_type=$3
    
    local normalization_applied=$(get_yaml_value "$yaml_file" "preprocessing.normalization_applied")
    local scaler_info=$(get_yaml_value "$yaml_file" "preprocessing.scaler")
    
    if [[ "$normalization_applied" == "True" || "$normalization_applied" == "true" ]]; then
        if [[ "$scaler_info" == "null" ]]; then
            log_error "$router/$model_type: normalization_applied is true but scaler info is missing"
        else
            # Check if scaler file exists
            local scaler_file="${ML_MODELS_DIR}/${router}/${model_type}/scaler_v1.0.0.pkl"
            if ! check_file_exists "$scaler_file" "Scaler file for $router/$model_type"; then
                log_error "$router/$model_type: Scaler file required but not found"
            fi
            
            # Validate scaler structure
            local scaler_type=$(get_yaml_value "$yaml_file" "preprocessing.scaler.type")
            if [[ "$scaler_type" == "null" ]]; then
                log_error "$router/$model_type: Missing preprocessing.scaler.type"
            fi
        fi
    elif [[ "$normalization_applied" == "False" || "$normalization_applied" == "false" ]]; then
        if [[ "$scaler_info" != "null" ]]; then
            log_warning "$router/$model_type: normalization_applied is false but scaler info is present"
        fi
        
        # Check if scaler file exists when it shouldn't
        local scaler_file="${ML_MODELS_DIR}/${router}/${model_type}/scaler_v1.0.0.pkl"
        if [[ -f "$scaler_file" ]]; then
            log_warning "$router/$model_type: Scaler file exists but normalization_applied is false"
        fi
    else
        log_error "$router/$model_type: preprocessing.normalization_applied must be true/false"
    fi
}

# Function to validate performance metrics
validate_performance_metrics() {
    local yaml_file=$1
    local router=$2
    local model_type=$3
    
    local required_metrics=("mae" "mape" "mse" "r2" "smape")
    
    for metric in "${required_metrics[@]}"; do
        local value=$(get_yaml_value "$yaml_file" "testing_performance_metrics.$metric")
        if [[ "$value" == "null" ]]; then
            log_error "$router/$model_type: Missing performance metric: $metric"
        fi
    done
}

# Function to validate model loading and parameters
validate_model_loading() {
    local yaml_file=$1
    local router=$2
    local model_type=$3
    
    log_info "Validating model loading and parameters for $router/$model_type"
    
    local model_dir="${ML_MODELS_DIR}/${router}/${model_type}"
    local model_file="${model_dir}/model_v1.0.0.pkl"
    local scaler_file="${model_dir}/scaler_v1.0.0.pkl"
    
    # Check if model file exists
    if [[ ! -f "$model_file" ]]; then
        log_error "$router/$model_type: Model file not found for validation"
        return
    fi
    
    # Create Python script to validate model
    python3 -c "
import pickle
import yaml
import sys
import os
import numpy as np
from datetime import datetime

def load_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f'ERROR: Failed to load YAML: {e}')
        sys.exit(1)

def load_model(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f'ERROR: Failed to load model: {e}')
        sys.exit(1)

def load_scaler(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        print(f'WARNING: Failed to load scaler: {e}')
        return None

# Load metadata
metadata = load_yaml('$yaml_file')
model = load_model('$model_file')

print('SUCCESS: Model loaded successfully')

# Validate model type
model_type_metadata = metadata.get('model', {}).get('type', '')
model_class_name = model.__class__.__name__

# Map model types to expected class names
expected_classes = {
    'linear_regression': ['LinearRegression'],
    'polynomial_regression': ['Pipeline'],  # Polynomial uses Pipeline
    'random_forest': ['RandomForestRegressor'],
    'deep_neural_network': ['MLPRegressor', 'Sequential', 'Model']  # scikit-learn or tensorflow/keras
}

expected_for_type = expected_classes.get(model_type_metadata, [])
if expected_for_type and model_class_name not in expected_for_type:
    print(f'WARNING: Model class {model_class_name} may not match expected type {model_type_metadata}')
else:
    print(f'SUCCESS: Model class {model_class_name} matches expected type {model_type_metadata}')

# Validate input features
input_features = metadata.get('features', {}).get('input', [])
expected_feature_count = len(input_features)

print(f'INFO: Expected input features ({expected_feature_count}): {input_features}')

# Try to determine model's expected input size
try:
    if hasattr(model, 'n_features_in_'):
        actual_feature_count = model.n_features_in_
        print(f'INFO: Model expects {actual_feature_count} input features')
        
        if actual_feature_count != expected_feature_count:
            print(f'ERROR: Feature count mismatch - metadata: {expected_feature_count}, model: {actual_feature_count}')
        else:
            print('SUCCESS: Feature count matches between metadata and model')
    elif hasattr(model, 'coef_'):
        # For linear models
        if hasattr(model.coef_, 'shape'):
            actual_feature_count = model.coef_.shape[0] if len(model.coef_.shape) == 1 else model.coef_.shape[1]
            print(f'INFO: Model expects {actual_feature_count} input features (from coef_)')
            
            if actual_feature_count != expected_feature_count:
                print(f'ERROR: Feature count mismatch - metadata: {expected_feature_count}, model: {actual_feature_count}')
            else:
                print('SUCCESS: Feature count matches between metadata and model')
    elif hasattr(model, 'named_steps'):
        # For pipeline models (polynomial regression)
        steps = model.named_steps
        if 'polynomialfeatures' in steps:
            poly_step = steps['polynomialfeatures']
            if hasattr(poly_step, 'n_features_in_'):
                actual_feature_count = poly_step.n_features_in_
                print(f'INFO: Pipeline expects {actual_feature_count} input features')
                
                if actual_feature_count != expected_feature_count:
                    print(f'ERROR: Feature count mismatch - metadata: {expected_feature_count}, model: {actual_feature_count}')
                else:
                    print('SUCCESS: Feature count matches between metadata and model')
        print('INFO: Pipeline model structure validated')
    else:
        print('WARNING: Cannot determine expected input feature count from model')
except Exception as e:
    print(f'WARNING: Error checking input features: {e}')

# Test model prediction with dummy data
try:
    # Create dummy input data
    dummy_input = np.random.rand(1, expected_feature_count)
    
    # Load scaler if needed
    normalization_applied = metadata.get('preprocessing', {}).get('normalization_applied', False)
    if normalization_applied:
        scaler_file_path = '$scaler_file'
        if os.path.exists(scaler_file_path):
            scaler = load_scaler(scaler_file_path)
            if scaler is not None:
                try:
                    dummy_input_scaled = scaler.transform(dummy_input)
                    prediction = model.predict(dummy_input_scaled)
                    print('SUCCESS: Model prediction with scaler successful')
                except Exception as e:
                    print(f'ERROR: Model prediction with scaler failed: {e}')
            else:
                print('WARNING: Scaler required but could not be loaded')
        else:
            print('ERROR: Scaler file not found but normalization_applied is true')
    else:
        try:
            prediction = model.predict(dummy_input)
            print('SUCCESS: Model prediction without scaler successful')
        except Exception as e:
            print(f'ERROR: Model prediction failed: {e}')
            
except Exception as e:
    print(f'WARNING: Could not test model prediction: {e}')

# Validate hyperparameters if available
try:
    hyperparams_metadata = metadata.get('model', {}).get('hyperparameters', {})
    
    if hyperparams_metadata:
        print('INFO: Validating hyperparameters...')
        
        # Common hyperparameters to check
        param_mappings = {
            'n_estimators': 'n_estimators',
            'max_depth': 'max_depth',
            'random_state': 'random_state',
            'degree': 'degree',
            'hidden_layer_sizes': 'hidden_layer_sizes',
            'max_iter': 'max_iter',
            'alpha': 'alpha'
        }
        
        for meta_param, model_param in param_mappings.items():
            if meta_param in hyperparams_metadata:
                expected_value = hyperparams_metadata[meta_param]
                
                # For pipeline models, check nested parameters
                if hasattr(model, 'named_steps'):
                    for step_name, step in model.named_steps.items():
                        if hasattr(step, model_param):
                            actual_value = getattr(step, model_param)
                            if actual_value != expected_value:
                                print(f'WARNING: Hyperparameter {meta_param} mismatch in {step_name} - metadata: {expected_value}, model: {actual_value}')
                            else:
                                print(f'SUCCESS: Hyperparameter {meta_param} matches in {step_name}')
                            break
                elif hasattr(model, model_param):
                    actual_value = getattr(model, model_param)
                    if actual_value != expected_value:
                        print(f'WARNING: Hyperparameter {meta_param} mismatch - metadata: {expected_value}, model: {actual_value}')
                    else:
                        print(f'SUCCESS: Hyperparameter {meta_param} matches')
                        
        print('INFO: Hyperparameter validation completed')
    else:
        print('INFO: No hyperparameters to validate in metadata')
        
except Exception as e:
    print(f'WARNING: Error validating hyperparameters: {e}')

print('INFO: Model validation completed')
" 2>/dev/null | while IFS= read -r line; do
        if [[ "$line" == ERROR:* ]]; then
            log_error "$router/$model_type: ${line#ERROR: }"
        elif [[ "$line" == WARNING:* ]]; then
            log_warning "$router/$model_type: ${line#WARNING: }"
        elif [[ "$line" == SUCCESS:* ]]; then
            log_success "$router/$model_type: ${line#SUCCESS: }"
        elif [[ "$line" == INFO:* ]]; then
            log_info "$router/$model_type: ${line#INFO: }"
        fi
    done
}

# Function to validate model consistency between routers
validate_cross_router_consistency() {
    local model_type=$1
    
    log_info "Validating cross-router consistency for $model_type"
    
    local routers=($(ls "$ML_MODELS_DIR"))
    local reference_router=""
    local reference_metadata=""
    
    # Find first router with this model type as reference
    for router in "${routers[@]}"; do
        local metadata_file="${ML_MODELS_DIR}/${router}/${model_type}/metadata_v1.0.0.yaml"
        if [[ -f "$metadata_file" ]]; then
            reference_router="$router"
            reference_metadata="$metadata_file"
            break
        fi
    done
    
    if [[ -z "$reference_router" ]]; then
        log_warning "No reference metadata found for model type: $model_type"
        return
    fi
    
    # Get reference values
    local ref_input_features=$(get_yaml_value "$reference_metadata" "features.input")
    local ref_output_feature=$(get_yaml_value "$reference_metadata" "features.output")
    local ref_model_version=$(get_yaml_value "$reference_metadata" "model.version")
    
    # Compare with other routers
    for router in "${routers[@]}"; do
        if [[ "$router" == "$reference_router" ]]; then
            continue
        fi
        
        local metadata_file="${ML_MODELS_DIR}/${router}/${model_type}/metadata_v1.0.0.yaml"
        if [[ ! -f "$metadata_file" ]]; then
            log_warning "$router: Missing $model_type model"
            continue
        fi
        
        # Compare features
        local input_features=$(get_yaml_value "$metadata_file" "features.input")
        local output_feature=$(get_yaml_value "$metadata_file" "features.output")
        local model_version=$(get_yaml_value "$metadata_file" "model.version")
        
        if [[ "$input_features" != "$ref_input_features" ]]; then
            log_warning "$router/$model_type: Input features differ from reference ($reference_router)"
        fi
        
        if [[ "$output_feature" != "$ref_output_feature" ]]; then
            log_warning "$router/$model_type: Output feature differs from reference ($reference_router)"
        fi
        
        if [[ "$model_version" != "$ref_model_version" ]]; then
            log_warning "$router/$model_type: Model version differs from reference ($reference_router)"
        fi
    done
}

# Function to validate file structure
validate_file_structure() {
    local router=$1
    local model_type=$2
    
    log_info "Validating file structure for $router/$model_type"
    
    local model_dir="${ML_MODELS_DIR}/${router}/${model_type}"
    local metadata_file="${model_dir}/metadata_v1.0.0.yaml"
    local model_file="${model_dir}/model_v1.0.0.pkl"
    
    # Check if directory exists
    if [[ ! -d "$model_dir" ]]; then
        log_error "Model directory not found: $model_dir"
        return
    fi
    
    # Check required files
    check_file_exists "$metadata_file" "Metadata file for $router/$model_type"
    check_file_exists "$model_file" "Model file for $router/$model_type"
    
    # Validate YAML syntax
    if [[ -f "$metadata_file" ]]; then
        validate_yaml_syntax "$metadata_file"
        validate_metadata_structure "$metadata_file" "$router" "$model_type"
        
        # Validate model loading and parameters
        if [[ -f "$model_file" ]]; then
            validate_model_loading "$metadata_file" "$router" "$model_type"
        fi
    fi
}

# Main validation function
main() {
    print_color "$CYAN" "╔════════════════════════════════════════════════════════════════════════════════╗"
    print_color "$CYAN" "║                           ML MODELS VALIDATION SCRIPT                           ║"
    print_color "$CYAN" "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo
    
    # Check if ML models directory exists
    if [[ ! -d "$ML_MODELS_DIR" ]]; then
        log_error "ML models directory not found: $ML_MODELS_DIR"
        exit 1
    fi
    
    log_info "Starting validation of ML models in: $ML_MODELS_DIR"
    echo
    
    # Discover available routers and models
    local routers=($(ls "$ML_MODELS_DIR" 2>/dev/null | sort))
    local all_model_types=()
    
    # Collect all unique model types
    for router in "${routers[@]}"; do
        if [[ -d "${ML_MODELS_DIR}/${router}" ]]; then
            local model_types=($(ls "${ML_MODELS_DIR}/${router}" 2>/dev/null | sort))
            for model_type in "${model_types[@]}"; do
                if [[ -d "${ML_MODELS_DIR}/${router}/${model_type}" ]]; then
                    if [[ ! " ${all_model_types[*]} " =~ " ${model_type} " ]]; then
                        all_model_types+=("$model_type")
                    fi
                fi
            done
        fi
    done
    
    # Print discovered structure
    print_color "$PURPLE" "╔════════════════════════════════════════════════════════════════════════════════╗"
    print_color "$PURPLE" "║                        DISCOVERED ML MODELS STRUCTURE                         ║"
    print_color "$PURPLE" "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo
    
    print_color "$CYAN" "Routers discovered:"
    for router in "${routers[@]}"; do
        echo "   • $router"
    done
    echo
    
    print_color "$CYAN" "Model types discovered:"
    for model_type in "${all_model_types[@]}"; do
        echo "   • $model_type"
    done
    echo
    
    # Print matrix of available models
    print_color "$CYAN" "Model availability matrix:"
    printf "%-25s" "Router/Model"
    for model_type in "${all_model_types[@]}"; do
        printf "%-20s" "$model_type"
    done
    echo
    printf "%-25s" "────────────────────────"
    for model_type in "${all_model_types[@]}"; do
        printf "%-20s" "──────────────────"
    done
    echo
    
    for router in "${routers[@]}"; do
        printf "%-25s" "$router"
        for model_type in "${all_model_types[@]}"; do
            local metadata_file="${ML_MODELS_DIR}/${router}/${model_type}/metadata_v1.0.0.yaml"
            if [[ -f "$metadata_file" ]]; then
                printf "%-20s" "Available"
            else
                printf "%-20s" "Missing"
            fi
        done
        echo
    done
    echo
    
    # Validate each router/model combination
    print_color "$PURPLE" "╔════════════════════════════════════════════════════════════════════════════════╗"
    print_color "$PURPLE" "║                   VALIDATING FILE STRUCTURE, METADATA & MODELS                  ║"
    print_color "$PURPLE" "╚════════════════════════════════════════════════════════════════════════════════╝"
    for router in "${routers[@]}"; do
        for model_type in "${all_model_types[@]}"; do
            validate_file_structure "$router" "$model_type"
        done
    done
    echo
    
    # Validate cross-router consistency
    print_color "$PURPLE" "╔════════════════════════════════════════════════════════════════════════════════╗"
    print_color "$PURPLE" "║                        VALIDATING CROSS-ROUTER CONSISTENCY                      ║"
    print_color "$PURPLE" "╚════════════════════════════════════════════════════════════════════════════════╝"
    for model_type in "${all_model_types[@]}"; do
        validate_cross_router_consistency "$model_type"
    done
    echo
    
    # Print all metadata in pretty format
    print_color "$PURPLE" "╔════════════════════════════════════════════════════════════════════════════════╗"
    print_color "$PURPLE" "║                             ALL ML MODELS METADATA                              ║"
    print_color "$PURPLE" "╚════════════════════════════════════════════════════════════════════════════════╝"
    for router in "${routers[@]}"; do
        for model_type in "${all_model_types[@]}"; do
            local metadata_file="${ML_MODELS_DIR}/${router}/${model_type}/metadata_v1.0.0.yaml"
            if [[ -f "$metadata_file" ]]; then
                pretty_print_yaml "$metadata_file" "$router" "$model_type"
            fi
        done
    done
    
    # Print summary
    print_color "$CYAN" "╔════════════════════════════════════════════════════════════════════════════════╗"
    print_color "$CYAN" "║                               VALIDATION SUMMARY                                ║"
    print_color "$CYAN" "╚════════════════════════════════════════════════════════════════════════════════╝"
    echo
    
    # Print statistics
    local total_models=0
    local available_models=0
    
    for router in "${routers[@]}"; do
        for model_type in "${all_model_types[@]}"; do
            ((total_models++))
            local metadata_file="${ML_MODELS_DIR}/${router}/${model_type}/metadata_v1.0.0.yaml"
            if [[ -f "$metadata_file" ]]; then
                ((available_models++))
            fi
        done
    done
    
    print_color "$CYAN" "Statistics:"
    echo "   • Total possible models: $total_models"
    echo "   • Available models: $available_models"
    echo "   • Coverage: $(( available_models * 100 / total_models ))%"
    echo "   • Routers: ${#routers[@]}"
    echo "   • Model types: ${#all_model_types[@]}"
    echo
    
    if [[ $ERRORS -gt 0 ]]; then
        print_color "$RED" "ERRORS FOUND ($ERRORS):"
        for error in "${ERROR_LOG[@]}"; do
            echo "   • $error"
        done
        echo
    fi
    
    if [[ $WARNINGS -gt 0 ]]; then
        print_color "$YELLOW" "WARNINGS FOUND ($WARNINGS):"
        for warning in "${WARNING_LOG[@]}"; do
            echo "   • $warning"
        done
        echo
    fi
    
    if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
        print_color "$GREEN" "All validations passed! No errors or warnings found."
        print_color "$GREEN" "Your ML models are properly structured and consistent!"
    elif [[ $ERRORS -eq 0 ]]; then
        print_color "$YELLOW" "Validation completed with $WARNINGS warnings but no errors."
        print_color "$YELLOW" "Consider addressing the warnings for optimal setup."
    else
        print_color "$RED" "Validation failed with $ERRORS errors and $WARNINGS warnings."
        print_color "$RED" "Please fix the errors before proceeding."
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    if ! command -v python3 &> /dev/null; then
        log_error "python3 is required but not installed"
        exit 1
    fi
    
    if ! python3 -c "import yaml" 2>/dev/null; then
        log_error "Python yaml module is required. Install with: pip install PyYAML"
        exit 1
    fi
    
    # Check for required ML packages
    local required_packages=("numpy" "sklearn")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import ${package}" 2>/dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        log_error "Required Python packages missing: ${missing_packages[*]}"
        log_error "Install with: pip install ${missing_packages[*]}"
        exit 1
    fi
    
    log_info "All required dependencies are available"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_dependencies
    main "$@"
fi
