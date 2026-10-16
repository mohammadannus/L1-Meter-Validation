# importing libraries

from __future__ import annotations

from models.base.base_model import BaseMeterModel
from models.registry import ModelRegistry


class ModelFactory:
    """
    Creates backend instances from the registry.
    """

    @staticmethod
    def create(model_name: str) -> BaseMeterModel:
        """
        Create a backend instance.

        Parameters
        ----------
        model_name : str
            Registered model name.

        Returns
        -------
        BaseMeterModel
            Backend instance.
        """

        backend_class = ModelRegistry.get_backend(model_name)

        return backend_class()