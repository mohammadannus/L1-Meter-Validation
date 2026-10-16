# importing libraries

from __future__ import annotations

from typing import Type

from models.base.base_model import BaseMeterModel


class ModelRegistry:
    """
    Central registry for all available model backends.

    Stores model backend classes without instantiating them.
    """

    _registry: dict[str, Type[BaseMeterModel]] = {}

    @classmethod
    def register(cls, model_name: str, backend_class: Type[BaseMeterModel]) -> None:
        """
        Register a backend class.

        Parameters
        ----------
        model_name : str
            Unique model identifier.

        backend_class : Type[BaseMeterModel]
            Backend class inheriting from BaseMeterModel.
        """

        if model_name in cls._registry and cls._registry[model_name] is not backend_class:
            import logging
            logging.getLogger(__name__).warning(f"Overwriting registered model backend for '{model_name}'.")

        cls._registry[model_name] = backend_class

    @classmethod
    def unregister(cls, model_name: str) -> None:
        """
        Remove a backend from the registry.
        """

        cls._registry.pop(model_name, None)

    @classmethod
    def get_backend(cls, model_name: str) -> Type[BaseMeterModel]:
        """
        Retrieve a backend class.
        """

        try:
            return cls._registry[model_name]

        except KeyError as error:
            available = ", ".join(cls.list_models()) or "None"

            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {available}"
            ) from error

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """
        Check whether a backend is registered.
        """

        return model_name in cls._registry

    @classmethod
    def list_models(cls) -> list[str]:
        """
        Return registered model names.
        """

        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Remove all registered models.
        Mainly useful for testing.
        """

        cls._registry.clear()