"""Tests for the vision processing module (API-only mode)."""

import time
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

from reachy_mini_conversation_app.vision.processors import (
    VisionConfig,
    VisionManager,
    VisionProcessor,
    initialize_vision_manager,
)


def test_vision_config_defaults() -> None:
    """Test VisionConfig has sensible defaults."""
    cfg = VisionConfig()
    assert cfg.vision_interval == 5.0
    assert cfg.max_new_tokens == 64
    assert cfg.jpeg_quality == 85
    assert cfg.max_retries == 3
    assert cfg.retry_delay == 1.0
    assert cfg.continuous_mode is False


def test_vision_config_custom_values() -> None:
    """Test VisionConfig accepts custom values."""
    cfg = VisionConfig(
        vision_interval=10.0,
        max_new_tokens=128,
        jpeg_quality=95,
        max_retries=5,
        retry_delay=2.0,
        continuous_mode=True,
    )
    assert cfg.vision_interval == 10.0
    assert cfg.max_new_tokens == 128
    assert cfg.jpeg_quality == 95
    assert cfg.max_retries == 5
    assert cfg.retry_delay == 2.0
    assert cfg.continuous_mode is True


def test_vision_processor_initialize() -> None:
    """Test VisionProcessor initializes in API-only mode."""
    processor = VisionProcessor()
    assert not processor._initialized

    with patch("reachy_mini_conversation_app.vision.processors.config"):
        result = processor.initialize()

    assert result is True
    assert processor._initialized


def test_vision_processor_process_image_not_initialized() -> None:
    """Test process_image returns error when not initialized."""
    processor = VisionProcessor()
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = processor.process_image(test_image)
    assert result == "Vision model not initialized"


def test_vision_processor_process_image_encode_failure() -> None:
    """Test process_image handles image encoding failure."""
    processor = VisionProcessor()
    with patch("reachy_mini_conversation_app.vision.processors.config"):
        processor.initialize()

    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (False, None)
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_image(test_image)

        assert result == "Failed to encode image"


def test_vision_processor_process_image_api_success() -> None:
    """Test process_image calls the API and returns the response."""
    processor = VisionProcessor()
    with patch("reachy_mini_conversation_app.vision.processors.config"):
        processor.initialize()

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "A baby playing with toys."

    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2, \
         patch("reachy_mini_conversation_app.vision.processors.config") as mock_config:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_config.LOCAL_LLM_URL = "http://localhost:11434/v1"
        mock_config.LOCAL_LLM_API_KEY = "test"
        mock_config.LOCAL_LLM_MODEL = "qwen2.5:3b"

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_cls.return_value = mock_client

            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = processor.process_image(test_image, "Describe this.")

            assert result == "A baby playing with toys."


def test_vision_processor_process_image_api_error() -> None:
    """Test process_image handles API errors gracefully."""
    processor = VisionProcessor()
    with patch("reachy_mini_conversation_app.vision.processors.config"):
        processor.initialize()

    with patch("reachy_mini_conversation_app.vision.processors.cv2") as mock_cv2, \
         patch("reachy_mini_conversation_app.vision.processors.config") as mock_config:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1
        mock_config.LOCAL_LLM_URL = "http://localhost:11434/v1"
        mock_config.LOCAL_LLM_API_KEY = "test"
        mock_config.LOCAL_LLM_MODEL = "qwen2.5:3b"

        with patch("openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Connection refused")
            mock_openai_cls.return_value = mock_client

            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = processor.process_image(test_image)

            assert "Vision Error" in result


def test_vision_processor_get_model_info() -> None:
    """Test get_model_info returns correct information."""
    processor = VisionProcessor()
    with patch("reachy_mini_conversation_app.vision.processors.config") as mock_config:
        mock_config.LOCAL_LLM_URL = "http://localhost:11434/v1"
        mock_config.LOCAL_LLM_MODEL = "qwen2.5:3b"
        processor.initialize()

        info = processor.get_model_info()

        assert info["initialized"] is True
        assert info["mode"] == "api"
        assert info["url"] == "http://localhost:11434/v1"
        assert info["model"] == "qwen2.5:3b"


@pytest.fixture
def mock_camera() -> Mock:
    """Create a mock camera object."""
    camera = Mock()
    camera.get_latest_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    return camera


def test_vision_manager_initialization(mock_camera: Mock) -> None:
    """Test VisionManager initializes successfully."""
    with patch("reachy_mini_conversation_app.vision.processors.config"):
        cfg = VisionConfig(vision_interval=2.0)
        manager = VisionManager(mock_camera, cfg)

        assert manager.vision_interval == 2.0
        assert manager.processor._initialized


def test_vision_manager_start_stop(mock_camera: Mock) -> None:
    """Test VisionManager can start and stop."""
    with patch("reachy_mini_conversation_app.vision.processors.config"):
        cfg = VisionConfig(continuous_mode=True)
        manager = VisionManager(mock_camera, cfg)

        manager.start()
        assert manager._thread is not None
        assert manager._thread.is_alive()

        time.sleep(0.1)

        manager.stop()
        assert manager._stop_event.is_set()
        assert not manager._thread.is_alive()


def test_vision_manager_on_demand_mode(mock_camera: Mock) -> None:
    """Test VisionManager in on-demand mode (continuous_mode=False) returns immediately."""
    with patch("reachy_mini_conversation_app.vision.processors.config"):
        cfg = VisionConfig(continuous_mode=False)
        manager = VisionManager(mock_camera, cfg)

        manager.start()
        time.sleep(0.1)
        manager.stop()

        # Camera should NOT have been called (on-demand mode exits the loop immediately)
        assert mock_camera.get_latest_frame.call_count == 0


def test_initialize_vision_manager_success(mock_camera: Mock) -> None:
    """Test initialize_vision_manager creates VisionManager successfully."""
    with patch("reachy_mini_conversation_app.vision.processors.config"):
        result = initialize_vision_manager(mock_camera)

        assert result is not None
        assert isinstance(result, VisionManager)


def test_initialize_vision_manager_failure(mock_camera: Mock) -> None:
    """Test initialize_vision_manager handles failure gracefully."""
    with patch("reachy_mini_conversation_app.vision.processors.config"), \
         patch.object(VisionProcessor, "initialize", return_value=False):
        result = initialize_vision_manager(mock_camera)

        assert result is None
