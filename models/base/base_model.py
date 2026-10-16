# importing libraries

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseMeterModel(ABC):
    """
    Abstract base class for every meter serial number model.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.
        """
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """
        Release model resources.
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image_path: Path) -> Any:
        """
        Convert an image into model inputs.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, processed_input: Any) -> Any:
        """
        Run inference.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, prediction: Any) -> dict[str, Any]:
        """
        Convert raw model output into a common response format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Return model metadata.
        """
        raise NotImplementedError