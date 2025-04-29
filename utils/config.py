import yaml
import os

def load_config(config_path):
    """
    Load configuration from a YAML file
    
    Args:
        config_path (str): Path to the configuration YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Validate configuration (basic checks)
    required_sections = ['data', 'model', 'training', 'generation']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate specific required parameters
    validate_config(config)
    
    return config

def validate_config(config):
    """
    Validate that all required parameters are present in the configuration
    
    Args:
        config (dict): Configuration dictionary
    """
    # Required parameters for each section
    required_params = {
        'data': ['file_path'],
        'model': ['block_size', 'd_model', 'num_heads', 'num_layers', 'd_ff', 'dropout'],
        'training': [
            'seed', 'batch_size', 'learning_rate', 'max_epochs', 
            'eval_interval', 'grad_clip'
        ],
        'generation': ['eval_examples', 'max_new_tokens', 'temperature', 'top_k']
    }
    
    # Check each section
    for section, params in required_params.items():
        for param in params:
            if param not in config[section]:
                raise ValueError(f"Missing required parameter '{param}' in section '{section}'")