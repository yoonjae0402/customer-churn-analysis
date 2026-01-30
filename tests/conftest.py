"""
Pytest configuration and shared fixtures.
"""
import pytest
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def app_dir(project_root):
    """Return the app directory."""
    return project_root / "app"


@pytest.fixture(scope="session")
def models_dir(app_dir):
    """Return the models directory."""
    return app_dir / "models"


@pytest.fixture(scope="session")
def data_dir(app_dir):
    """Return the data directory."""
    return app_dir / "data"
