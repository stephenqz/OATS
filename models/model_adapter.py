from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from typing import Any
from torch.nn import Module
from transformers import PreTrainedTokenizerBase


"""
To add support for a new model, you need to create a new adapter class that inherits from ModelAdapter, and a new 
adapter class that inherits from LayerAdapter. 
"""


class LayerAdapter(ABC):
    """
    To implement a new layer adapter, implement the interface defined in this class
    """

    @property
    @abstractmethod
    def layer(self) -> Module:
        """
        Instance of the transformer layer to be wrapped. This contains the forward() method of the original model
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_args_position(self) -> int:
        """
        Returns the position of the hidden_states argument in the layer's forward method.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_states_output_position(self) -> int:
        """
        Returns the position of the hidden_states in the output of the layer's forward method.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def linear_layer_order(self) -> int:
        """
        Returns the order of linear layers in the transformer block.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def qkv_names(self) -> int:
        """
        Returns the names of the qkv modules in the transformer block.
        """
        raise NotImplementedError
    
    def get_updated_args(self, hidden_states: Any, args: tuple) -> tuple:
        """
        `args` is a tuple of the arguments to the layer's forward method. hidden_states is the new value for the
        hidden_states argument. This method returns a new tuple of arguments with the hidden_states argument updated.
        """
        return (
            args[: self.hidden_states_args_position] + (hidden_states,) + args[self.hidden_states_args_position + 1 :]
        )


class ModelAdapter(ABC):
    """
    To implement a new model adapter, implement the interface defined in this class
    """
    
    @property
    @abstractmethod
    def model(self) -> Module:
        """
        The original model that slicegpt interacts with.
        """
        raise NotImplementedError

    @abstractmethod
    def get_layers(self) -> Sequence[LayerAdapter]:
        """
        Returns a list of LayerAdapters, one for each layer in the model.
        """
        raise NotImplementedError

    @abstractmethod
    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        """
        Assigns the given layer to the model at the given index.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self) -> list[Module]:
        """
        Returns a list of the embedding modules in the model.
        """
        raise NotImplementedError

    @property
    def no_split_module_classes(self) -> list[str] | None:
        """
        A list of strings specifying the class names of modules that should not be split.
        See https://huggingface.co/docs/accelerate/concept_guides/big_model_inference for more details.
        """
        return [self.original_layer_type.__name__]

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        This method is called after the model is initialized and all the properties are set.
        Override in subclasses to perform any additional setup.
        """
        pass

    @classmethod
    def from_model(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        eval_model: bool = False,
        local_files_only: bool = False,
    ) -> ModelAdapter:
        """
        Create the model based on the given name path and return the corresponding ModelAdapter instance.
        Raise NotImplementedError if the model is not supported.
        Note: for this method to work the corresponding ModelAdapter subclass must be imported.

        Args:
            model_name: The name of the model, e.g. 'microsoft/phi-2'.
            model_path: The path to the model.
            dtype: The torch dtype to create the model with.
            local_files_only: Whether to only load local files (no attempt to download).
            token: The token to use for authentication.

        Returns:
            The corresponding ModelAdapter instance.
        """

        def find_recursively(adapter_cls: type[ModelAdapter]) -> ModelAdapter | None:
            """
            Recursively search for a subclass that can handle the model.
            """
            # depth first search to find the most specific subclass that can handle the model
            for subclass in adapter_cls.__subclasses__():
                candidate = find_recursively(subclass)
                if candidate is not None:
                    return candidate

            if inspect.isabstract(adapter_cls):
                return None

            return adapter_cls._from_model(
                model_name,
                model_path=model_path,
                dtype=dtype,
                eval_model= eval_model,
                local_files_only=local_files_only,
            )

        adapter = find_recursively(cls)
        if adapter is not None:
            return adapter
        
        adapter = find_recursively(cls)
        if adapter is not None:
            return adapter

        raise NotImplementedError(f"{model_path} is neither a Hugging Face model nor a supported local model.")

    @classmethod
    def _from_model(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        eval_model: bool = False,
        local_files_only: bool = False,
    ) -> ModelAdapter | None:
        return cls._from_pretrained(
            model_name,
            model_path=model_path,
            dtype=dtype,
            eval_model=eval_model,
            local_files_only=local_files_only,
            )

    @classmethod
    @abstractmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        eval_model: bool = False,
        local_files_only: bool = False,
    ) -> ModelAdapter | None:
        """
        Load the pretrained model from the given path and return a ModelAdapter instance.
        Return None if the model_name is not supported.
        See `from_model` for more details.
        """
        raise NotImplementedError