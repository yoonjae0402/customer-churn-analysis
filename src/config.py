
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        # Resolve config path relative to project root (assuming script runs from root or src is parallel)
        # Better: use an absolute path strategy if possible, or assume execution from project root
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Sensitive credentials from ENV
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            # If not found in CWD, try looking relative to this file
            fallback = Path(__file__).parent.parent / "config.yaml"
            if fallback.exists():
                return yaml.safe_load(fallback.open())
            raise FileNotFoundError(f"Config file not found at {self.config_path} or {fallback}")
        
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def paths(self):
        return self.config["paths"]

    @property
    def model_config(self):
        return self.config["model"]
    
    @property
    def training_config(self):
        return self.config["training"]
        
    @property
    def feature_config(self):
        return self.config["features"]

# Singleton instance
config = Config()
