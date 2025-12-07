"""Pytest configuration and shared fixtures for LLM tests."""
from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "benchmark: performance benchmark tests that measure actual latency",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that take a long time to run (load large models)",
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--device",
        action="store",
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to use for tests (cuda, mps, or cpu)",
    )
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow tests that load large models",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="--skip-slow option provided")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def test_device(request):
    """Get the device to use for tests from command line."""
    return request.config.getoption("--device")


@pytest.fixture
def sample_messages():
    """Sample message history for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "What is Python?"},
    ]


@pytest.fixture
def simple_prompts():
    """Simple test prompts for quick testing."""
    return [
        "Hello",
        "What is 2+2?",
        "Name a color",
    ]
