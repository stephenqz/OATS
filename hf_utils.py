import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from models.model_adapter import ModelAdapter

def get_model_and_tokenizer(
    model_name: str,
    model_path: str | None = None,
    *,
    dtype: torch.dtype = torch.float16,
    eval_model: bool = False,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    
    local_model = model_path is not None

    # for HF models the path to use is the model name
    if not local_model:
        # LLAMA VARIANTS
        if model_name == "llama3-8b":
            model_path = "meta-llama/Meta-Llama-3-8B"
        elif model_name == "llama3-70b":
            model_path = "meta-llama/Meta-Llama-3-70B"
        elif model_name == "phi-3-mini":
            model_path = "microsoft/Phi-3-mini-4k-instruct"
        elif model_name == "phi-3-medium":
            model_path = "microsoft/Phi-3-medium-128k-instruct"
    
    print(
        f"Loading %s config %s from %s",
        model_name,
        "and model weights",
        model_path if local_model else 'Hugging Face',
    )
    
    model_adapter = ModelAdapter.from_model(
        model_name,
        model_path=model_path,
        dtype=dtype,
        eval_model=eval_model,
        local_files_only=local_model,
    )

    model = model_adapter.model
    model.eval() 

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=local_model)

    model_adapter.post_init(tokenizer)
    print("Loading model done")

    return model_adapter, tokenizer