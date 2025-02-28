from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import torch

def distribute_model(model_adapter) -> None:
    """Distribute the model across available GPUs."""
    model = model_adapter.model
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=model_adapter.no_split_module_classes,
    )

    print(max_memory)

    print(model_adapter.no_split_module_classes)
    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=model_adapter.no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

def map_tensors(obj, device, dtype):
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj