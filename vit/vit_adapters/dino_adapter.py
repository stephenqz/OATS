from torch.nn import Module
from transformers.models.dinov2.modeling_dinov2 import Dinov2Layer, Dinov2ForImageClassification
from models.model_adapter import LayerAdapter

class DinoLayerAdapter(LayerAdapter):
    def __init__(self, layer: Dinov2Layer) -> None:
        super().__init__()
        self._layer: Dinov2Layer = layer

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
        return ['attention.attention.query', 'attention.attention.key', 'attention.attention.value', "attention.output.dense", 'mlp.weights_in', 'mlp.weights_out']
    
    @property
    def qkv_names(self) -> list[str]:
        return ['attention.attention.query', 'attention.attention.key', 'attention.attention.value']

    def get_qkv_partition(self) -> list[int]:
        return None


class DINOModelAdapter(object):
    def __init__(self, model_path, dtype):

        print("Loading model from: " + str(model_path))
        model = Dinov2ForImageClassification.from_pretrained(model_path, attn_implementation='eager')
        model.config.torch_dtype = dtype

        self._model: Dinov2ForImageClassification = model
        
    @property
    def model(self) -> Module:
        return self._model

    @property
    def layer_adapter_type(self) -> type:
        return DinoLayerAdapter
    
    @property
    def original_layer_type(self) -> type:
        return Dinov2Layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.dinov2.encoder.layer]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.dinov2.encoder.layer[index] = new_layer
