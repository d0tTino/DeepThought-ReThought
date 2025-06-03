# Configuration module for DeepThought reThought
import os
import yaml
import logging

logger = logging.getLogger(__name__)

# --- Default Settings ---
DEFAULTS = {
    "nats": {
        "url": "nats://localhost:4222",
        "stream_name": "deepthought_events",
    },
    "logging": {
        "level": "INFO",
    }
}

# --- Load Configuration from YAML and Environment Variables ---
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config.yml") # Relative to this file's location in src/deepthought/

def load_config_from_yaml(file_path: str) -> dict:
    """Loads configuration from a YAML file."""
    if not os.path.exists(file_path):
        logger.info(f"Configuration file not found at {file_path}. Using defaults and environment variables.")
        return {}
    try:
        with open(file_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {file_path}.")
        return yaml_config if yaml_config else {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {file_path}: {e}. Using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading configuration file {file_path}: {e}. Using defaults.")
        return {}

# Load initial config from YAML
_yaml_config = load_config_from_yaml(CONFIG_FILE_PATH)

# --- Configuration Access Functions ---
# Helper to merge dictionaries, giving priority to the first one (for overrides)
def _deep_merge_configs(primary_config: dict, fallback_config: dict) -> dict:
    merged = fallback_config.copy()
    for key, value in primary_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_configs(value, merged[key])
        else:
            merged[key] = value
    return merged

# Start with defaults, then merge YAML, then environment variables will override
_current_config = DEFAULTS.copy()
_current_config = _deep_merge_configs(_yaml_config, _current_config)


# --- Provide access to settings ---
# Example: NATS URL
# Order of precedence: Environment Variable > config.yml > Default
NATS_URL = os.getenv("DTR_NATS_URL", _current_config.get("nats", {}).get("url", DEFAULTS["nats"]["url"]))
NATS_STREAM_NAME = os.getenv("DTR_NATS_STREAM_NAME", _current_config.get("nats", {}).get("stream_name", DEFAULTS["nats"]["stream_name"]))

# Example: Logging Level
LOGGING_LEVEL_STR = os.getenv("DTR_LOGGING_LEVEL", _current_config.get("logging", {}).get("level", DEFAULTS["logging"]["level"])).upper()

# Convert string level to logging module's level constants
LOGGING_LEVEL = getattr(logging, LOGGING_LEVEL_STR, logging.INFO)
if not isinstance(LOGGING_LEVEL, int): # Check if getattr failed
    logger.warning(f"Invalid logging level string '{LOGGING_LEVEL_STR}'. Defaulting to INFO.")
    LOGGING_LEVEL = logging.INFO


# --- Global SETTINGS dictionary (optional, for direct access if preferred) ---
# This reflects the final merged configuration
SETTINGS = {
    "nats": {
        "url": NATS_URL,
        "stream_name": NATS_STREAM_NAME,
    },
    "logging": {
        "level_str": LOGGING_LEVEL_STR, # String representation
        "level": LOGGING_LEVEL,       # Actual logging level int
    }
}

# --- Helper functions to get specific settings (optional but good practice) ---
def get_nats_url() -> str:
    return NATS_URL

def get_nats_stream_name() -> str:
    return NATS_STREAM_NAME

def get_logging_level() -> int:
    return LOGGING_LEVEL

def get_logging_level_str() -> str:
    return LOGGING_LEVEL_STR

if __name__ == '__main__':
    # For testing the config loading
    print(f"--- Configuration Test ---")
    print(f"Config file used: {CONFIG_FILE_PATH if os.path.exists(CONFIG_FILE_PATH) else 'Not found, using defaults.'}")
    print(f"NATS URL: {get_nats_url()}")
    print(f"NATS Stream Name: {get_nats_stream_name()}")
    print(f"Logging Level String: {get_logging_level_str()}")
    print(f"Logging Level (int): {get_logging_level()}")
    print(f"""
Full SETTINGS dictionary:
{SETTINGS}""")
