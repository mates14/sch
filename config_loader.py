#!/usr/bin/python3
"""
Configuration loader for RTS2 scheduler.
Loads configuration from YAML file and provides access to configuration values.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass

class Config:
    """Configuration loader and accessor for the RTS2 scheduler."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(self.config_file):
                raise ConfigurationError(f"Configuration file {self.config_file} not found")
                
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if not config:
                raise ConfigurationError(f"Empty or invalid configuration in {self.config_file}")
                
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML in {self.config_file}: {e}")
    
    def _validate_config(self):
        """Validate the configuration structure."""
        required_sections = ['resources', 'scheduler']
        
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"Missing required section '{section}' in configuration")
        
        # Validate resources
        if not self.config['resources']:
            raise ConfigurationError("No resources defined in configuration")
            
        for resource_name, resource in self.config['resources'].items():
            for key in ['name', 'location', 'database', 'rts2_json']:
                if key not in resource:
                    raise ConfigurationError(f"Missing '{key}' in resource '{resource_name}'")
            
            # Validate location
            for key in ['latitude', 'longitude', 'elevation']:
                if key not in resource['location']:
                    raise ConfigurationError(f"Missing '{key}' in location for resource '{resource_name}'")
            
            # Validate database
            for key in ['dbname', 'user', 'password', 'host']:
                if key not in resource['database']:
                    raise ConfigurationError(f"Missing '{key}' in database for resource '{resource_name}'")
            
            # Validate RTS2 JSON
            for key in ['url', 'user', 'password']:
                if key not in resource['rts2_json']:
                    raise ConfigurationError(f"Missing '{key}' in RTS2 JSON for resource '{resource_name}'")
        
        # Validate scheduler
        scheduler_config = self.config['scheduler']
        for key in ['slice_size', 'horizon_file', 'min_altitude', 'timelimit', 'mip_gap']:
            if key not in scheduler_config:
                raise ConfigurationError(f"Missing '{key}' in scheduler configuration")
    
    def get_resources(self) -> Dict[str, Any]:
        """Get resources configuration."""
        return self.config['resources']
    
    def get_resource_db_config(self, resource_name: str) -> Optional[Dict[str, Any]]:
        """Get database configuration for a specific resource."""
        resource = self.config['resources'].get(resource_name)
        if not resource or 'database' not in resource:
            return None
        return resource['database']
    
    def get_resource_rts2_config(self, resource_name: str) -> Optional[Dict[str, Any]]:
        """Get RTS2 JSON configuration for a specific resource."""
        resource = self.config['resources'].get(resource_name)
        if not resource or 'rts2_json' not in resource:
            return None
        return resource['rts2_json']
    
    def get_scheduler_param(self, param_name: str, default: Any = None) -> Any:
        """Get a scheduler parameter."""
        return self.config['scheduler'].get(param_name, default)
    
    def get_output_path(self, path_name: str, default: str = '.') -> str:
        """Get an output path."""
        if 'output' not in self.config:
            return default
            
        return self.config['output'].get(path_name, default)
    
    def get_rts2_config_map(self) -> Dict[str, Dict[str, str]]:
        """Get a mapping of resource names to their RTS2 configurations."""
        result = {}
        for resource_name in self.config['resources']:
            rts2_config = self.get_resource_rts2_config(resource_name)
            if rts2_config:
                result[resource_name] = rts2_config
        return result
    
    def __str__(self) -> str:
        """String representation of the configuration (with masked passwords)."""
        # Create a copy of the config to avoid modifying the original
        masked_config = yaml.safe_load(yaml.dump(self.config))
        
        # Mask passwords in resources
        for resource_name, resource in masked_config.get('resources', {}).items():
            if 'database' in resource and 'password' in resource['database']:
                resource['database']['password'] = '********'
            if 'rts2_json' in resource and 'password' in resource['rts2_json']:
                resource['rts2_json']['password'] = '********'
        
        return yaml.dump(masked_config, default_flow_style=False)
