import torch
from torch.nn import Module
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3ForCausalLM

from models.model_adapter import LayerAdapter, ModelAdapter


class Phi3LayerAdapter(LayerAdapter):
    def __init__(self, layer: Phi3DecoderLayer) -> None:
        super().__init__()
        self._layer: Phi3DecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    @property
    def linear_layer_order(self) -> list[str]:
        return ["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"]
    
    @property
    def qkv_names(self) -> list[str]:
        return ["self_attn.qkv_proj"]

    def get_qkv_partition(self) -> list[int]:
        query_dim = self.layer.self_attn.num_heads * self.layer.self_attn.head_dim
        key_dim = self.layer.self_attn.num_heads * self.layer.self_attn.head_dim + self.layer.self_attn.num_key_value_heads * self.layer.self_attn.head_dim
        return [query_dim, key_dim]


class Phi3ModelAdapter(ModelAdapter):
    def __init__(self, model: Phi3ForCausalLM) -> None:
        super().__init__()
        self._model: Phi3ForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model
    
    @property
    def layer_adapter_type(self) -> type:
        return Phi3LayerAdapter
    
    @property
    def original_layer_type(self) -> type:
        return Phi3DecoderLayer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        eval_model: bool =False,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("phi-3"):
            return None

        if eval_model:
            model = Phi3ForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only, attn_implementation="flash_attention_2", device_map="auto"
            )
        else:
            model = Phi3ForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only, attn_implementation="flash_attention_2"
            )
        model.config.torch_dtype = dtype
        return Phi3ModelAdapter(model)