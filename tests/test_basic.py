# tests/test_basic.py
import pytest
import os
import yaml
import tempfile
import shutil # For cleaning up temp directory
import importlib # For reloading app_config

# Existing basic tests
def test_example_basic_assertion():
    """A basic test that always passes, confirming test setup works."""
    assert True, "This basic assertion should always pass."

# Example of importing a module (original)
# We will use app_config alias for the new tests for clarity
try:
    from src.deepthought import config
except ImportError:
    config = None # Handle if module cannot be imported

def test_import_config():
    """Tests if a core configuration module can be imported (original test)."""
    assert config is not None, "Failed to import config module from src.deepthought (original test). Check PYTHONPATH and src/deepthought/config.py."

# --- New tests for configuration loading ---
from src.deepthought import config as app_config # Use an alias for new tests

# Store original environ for restoration
_original_environ = None

def setup_module(module):
    """Save original environment variables before tests in this module."""
    global _original_environ
    _original_environ = os.environ.copy()
    # Also store original config path from app_config for the module
    module._original_app_config_path_for_module = app_config.CONFIG_FILE_PATH


def teardown_module(module):
    """Restore original environment variables after all tests in this module."""
    global _original_environ
    if _original_environ is not None:
        os.environ.clear()
        os.environ.update(_original_environ)
    # Restore original config path for the module
    if hasattr(module, '_original_app_config_path_for_module'):
        app_config.CONFIG_FILE_PATH = module._original_app_config_path_for_module
        reload_app_config(app_config)


def reload_app_config(config_module_to_reload):
    """Helper to reload the app_config module to pick up new env vars or file changes."""
    importlib.reload(config_module_to_reload)

@pytest.fixture
def temp_config_fixture(request):
    """Create a temporary directory and manage app_config.CONFIG_FILE_PATH for config.yml tests."""
    temp_project_root = tempfile.mkdtemp()

    # Store the original CONFIG_FILE_PATH from app_config at the start of the fixture
    original_app_config_path_in_fixture = app_config.CONFIG_FILE_PATH

    # New temporary config file path
    temp_config_file = os.path.join(temp_project_root, "config.yml")

    # Temporarily set the app_config's path to our temp file
    app_config.CONFIG_FILE_PATH = temp_config_file
    reload_app_config(app_config) # Reload to make config.py use the new path

    def finalizer():
        shutil.rmtree(temp_project_root)
        # Restore original path and reload again
        app_config.CONFIG_FILE_PATH = original_app_config_path_in_fixture
        reload_app_config(app_config)

    request.addfinalizer(finalizer)
    # Return the path to the temporary config file itself, so tests can create/modify it
    return temp_config_file

def test_load_defaults_when_no_file_no_env(temp_config_fixture):
    """Test that defaults are loaded if config.yml doesn't exist and no env vars are set."""
    os.environ.pop("DTR_NATS_URL", None)
    os.environ.pop("DTR_NATS_STREAM_NAME", None)
    os.environ.pop("DTR_LOGGING_LEVEL", None)

    # temp_config_fixture provides the path to where config.yml *would* be. Ensure it's not there.
    if os.path.exists(temp_config_fixture): # temp_config_fixture is the path to config.yml
        os.remove(temp_config_fixture)

    reload_app_config(app_config)

    assert app_config.get_nats_url() == app_config.DEFAULTS["nats"]["url"]
    assert app_config.get_nats_stream_name() == app_config.DEFAULTS["nats"]["stream_name"]
    assert app_config.get_logging_level_str() == app_config.DEFAULTS["logging"]["level"]

def test_load_from_yaml_file(temp_config_fixture):
    """Test that settings are loaded from config.yml."""
    os.environ.pop("DTR_NATS_URL", None)
    os.environ.pop("DTR_NATS_STREAM_NAME", None)
    os.environ.pop("DTR_LOGGING_LEVEL", None)

    test_yaml_content = {
        "nats": {"url": "nats://custom_yaml_url:4222", "stream_name": "yaml_stream"},
        "logging": {"level": "DEBUG_YAML"}
    }
    # temp_config_fixture is the path to config.yml
    with open(temp_config_fixture, 'w') as f:
        yaml.dump(test_yaml_content, f)

    reload_app_config(app_config)

    assert app_config.get_nats_url() == "nats://custom_yaml_url:4222"
    assert app_config.get_nats_stream_name() == "yaml_stream"
    assert app_config.get_logging_level_str() == "DEBUG_YAML"

def test_env_vars_override_yaml_and_defaults(temp_config_fixture):
    """Test that environment variables override config.yml and defaults."""
    os.environ["DTR_NATS_URL"] = "nats://env_url:4222"
    os.environ["DTR_NATS_STREAM_NAME"] = "env_stream"
    os.environ["DTR_LOGGING_LEVEL"] = "WARNING_ENV"

    test_yaml_content = {
        "nats": {"url": "nats://yaml_url_should_be_overridden:4222", "stream_name": "yaml_stream_overridden"},
        "logging": {"level": "DEBUG_YAML_SHOULD_BE_OVERRIDDEN"}
    }
    with open(temp_config_fixture, 'w') as f:
        yaml.dump(test_yaml_content, f)

    reload_app_config(app_config)

    assert app_config.get_nats_url() == "nats://env_url:4222"
    assert app_config.get_nats_stream_name() == "env_stream"
    assert app_config.get_logging_level_str() == "WARNING_ENV"

def test_partial_env_override(temp_config_fixture):
    """Test that an env var for one setting doesn't wipe out other settings from YAML/defaults."""
    os.environ["DTR_NATS_URL"] = "nats://partial_env_override_url:4222"
    os.environ.pop("DTR_NATS_STREAM_NAME", None) # Ensure stream name comes from YAML
    os.environ.pop("DTR_LOGGING_LEVEL", None)    # Ensure logging level comes from YAML

    test_yaml_content = {
        "nats": {
            "url": "nats://yaml_url_should_be_overridden:4222", # This will be overridden by DTR_NATS_URL
            "stream_name": "yaml_stream_should_be_used"      # This should be used
        },
        "logging": {"level": "INFO_FROM_YAML"} # This should be used
    }
    with open(temp_config_fixture, 'w') as f:
        yaml.dump(test_yaml_content, f)

    reload_app_config(app_config)

    assert app_config.get_nats_url() == "nats://partial_env_override_url:4222"
    assert app_config.get_nats_stream_name() == "yaml_stream_should_be_used"
    assert app_config.get_logging_level_str() == "INFO_FROM_YAML"

def test_invalid_logging_level_in_config_falls_back_to_default_actual_level(temp_config_fixture):
    """Test that an invalid logging level string in YAML (not overridden by ENV) uses default for actual level."""
    os.environ.pop("DTR_NATS_URL", None)
    os.environ.pop("DTR_NATS_STREAM_NAME", None)
    os.environ.pop("DTR_LOGGING_LEVEL", None)

    test_yaml_content = {"logging": {"level": "VERY_WRONG_LEVEL"}}
    with open(temp_config_fixture, 'w') as f:
        yaml.dump(test_yaml_content, f)

    reload_app_config(app_config)

    import logging # for the constant
    # The string representation should reflect what was set, if it came from env or yaml
    assert app_config.get_logging_level_str() == "VERY_WRONG_LEVEL"
    # The actual integer level should fall back to default logging.INFO
    assert app_config.get_logging_level() == logging.INFO

def test_default_stream_name_if_not_in_yaml_or_env(temp_config_fixture):
    """Test that default stream name is used if not in YAML or ENV."""
    os.environ.pop("DTR_NATS_URL", None)
    os.environ.pop("DTR_NATS_STREAM_NAME", None) # Ensure not in ENV
    os.environ.pop("DTR_LOGGING_LEVEL", None)

    test_yaml_content = { # NATS section exists, but no stream_name
        "nats": {"url": "nats://minimal_yaml_url:4222"},
        "logging": {"level": "DEBUG"}
    }
    with open(temp_config_fixture, 'w') as f:
        yaml.dump(test_yaml_content, f)

    reload_app_config(app_config)

    assert app_config.get_nats_url() == "nats://minimal_yaml_url:4222" # From YAML
    assert app_config.get_nats_stream_name() == app_config.DEFAULTS["nats"]["stream_name"] # Should be default
    assert app_config.get_logging_level_str() == "DEBUG" # From YAML
