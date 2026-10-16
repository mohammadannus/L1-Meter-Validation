# importing libraries

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from huggingface_hub import snapshot_download


class ModelManager:
    """
    Generic model lifecycle manager.

    Responsibilities
    ----------------
    - Read model configuration.
    - Download models.
    - Locate local models.
    - Lazy load models.
    - Unload models.
    - GPU memory cleanup.

    This class is intentionally backend agnostic.
    It knows nothing about Qwen, InternVL, OCR,
    Hugging Face Transformers or any specific model.
    """

    def __init__(self, config_path: str = "configs/models.yaml") -> None:

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        self._loaded_model_name: str | None = None
        self._loaded_backend: Any = None
        self._loaded_resources: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """
        Return all configured model names.
        """

        return sorted(self.config.get("models", {}).keys())

    def get_model_config(self, model_name: str) -> dict[str, Any]:
        """
        Return configuration for a model.
        """

        models = self.config.get("models", {})

        if model_name not in models:
            available = ", ".join(self.list_models())

            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {available}"
            )

        return models[model_name]

    def get_model_path(self, model_name: str) -> Path:
        """
        Return local storage path.
        """

        config = self.get_model_config(model_name)

        return Path(config["local_dir"])

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def is_downloaded(self, model_name: str) -> bool:
        """
        Check whether a model exists locally.

        This is a lightweight existence check.
        """

        model_path = self.get_model_path(model_name)

        return model_path.exists() and any(model_path.iterdir())

    def download(self, model_name: str) -> Path:
        """
        Download a model if not already present.
        """

        model_config = self.get_model_config(model_name)

        model_path = self.get_model_path(model_name)

        if self.is_downloaded(model_name):

            print(f"[SKIP] {model_name} already exists.")

            return model_path
         
        print(f"[DOWNLOAD] {model_name}")

        snapshot_download(
            repo_id=model_config["repo_id"],
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )

        print(f"[DONE] {model_name}")

        return model_path

    def download_all(self) -> dict[str, bool]:
        """
        Download every configured model.

        Returns
        -------
        dict
            Download status for each model.
        """

        results: dict[str, bool] = {}

        for model_name in self.list_models():

            try:

                self.download(model_name)

                results[model_name] = True

            except Exception as error:

                print(f"[FAILED] {model_name}")

                print(error)

                results[model_name] = False

        return results

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def current_model(self) -> str | None:
        """
        Return currently loaded model.
        """

        return self._loaded_model_name

    def is_loaded(self, model_name: str) -> bool:
        """
        Check whether a model is already loaded.
        """

        return self._loaded_model_name == model_name

    def load(
        self,
        model_name: str,
        loader: Callable[[Path], tuple[Any, dict[str, Any]]],
    ) -> Any:
        """
        Lazy load a backend.

        Parameters
        ----------
        model_name
            Configured model name.

        loader
            Backend loader callable.

        Returns
        -------
        Backend instance.
        """

        if self.is_loaded(model_name):

            return self._loaded_backend

        self.unload()

        model_path = self.download(model_name)

        backend, resources = loader(model_path)

        self._loaded_model_name = model_name
        self._loaded_backend = backend
        self._loaded_resources = resources

        return backend

    # ------------------------------------------------------------------
    # Unloading
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """
        Release currently loaded backend and GPU memory.
        """

        self._loaded_backend = None
        self._loaded_resources.clear()
        self._loaded_model_name = None

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            torch.cuda.synchronize()