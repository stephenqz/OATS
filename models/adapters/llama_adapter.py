import torch
from torch.nn import Module
from transformers import PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

from models.model_adapter import LayerAdapter, ModelAdapter

class LlamaLayerAdapter(LayerAdapter):
    def __init__(self, layer: LlamaDecoderLayer) -> None:
        super().__init__()
        self._layer: LlamaDecoderLayer = layer

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
        return ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
    
    @property
    def qkv_names(self) -> list[str]:
        return ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']

    def get_qkv_partition(self) -> list[int]:
        return None


class LlamaModelAdapter(ModelAdapter):
    def __init__(self, model: LlamaForCausalLM) -> None:
        super().__init__()
        self._model: LlamaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def layer_adapter_type(self) -> type:
        return LlamaLayerAdapter
    
    @property
    def original_layer_type(self) -> type:
        return LlamaDecoderLayer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens, self.model.model.rotary_emb]
    
    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Set pad token for LLAMA 3
        tokenizer.pad_token = tokenizer.eos_token
        self._model.config.pad_token_id = tokenizer.pad_token_id
        print("Set padding token for LLAMA 3")

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        eval_model: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("llama"):
            return None
        
        print("Loading model from: " + str(model_path))
        if eval_model:
            model = LlamaForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only, device_map="auto"
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
            )
        model.config.torch_dtype = dtype

        return LlamaModelAdapter(model)
