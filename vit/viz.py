import torch

import cv2
from PIL import Image
import requests

from config import config
from OATS.pruning_utils import find_layers
from OATS.compressed_linear import CompressedLinear
import numpy as np

def viz_vit(model_adapter, split_part, image_processor, rollout_path) -> None:
    # List of URLs of images from COCO Dataset presented in paper. 
    url_paths = ['http://farm4.staticflickr.com/3739/9384480599_8353cbf7b6_z.jpg', \
             'http://images.cocodataset.org/val2017/000000039769.jpg', \
             'http://farm1.staticflickr.com/126/344514859_c0066b0469_z.jpg', \
             'http://farm6.staticflickr.com/5324/6945159772_ddd6653186_z.jpg', \
             'http://farm6.staticflickr.com/5245/5284494119_68887578ed_z.jpg', \
             'http://farm6.staticflickr.com/5519/9176127317_01d7585ddd_z.jpg',
            ]
    
    image_list = []
    for url in url_paths:
        image_list.append(Image.open(requests.get(url, stream=True).raw))
    
    rollout_inputs =[]
    for img in image_list:
        rollout_inputs.append(image_processor(images=img, return_tensors="pt")['pixel_values'])

    model = model_adapter.model
    model.to(config.device)

    split_model_(model_adapter, split_part)

    for ri_idx, ri in enumerate(rollout_inputs):
        attention_rollout = AttentionRollout(model, head_fusion="mean", discard_ratio=0.4)
        rollout_input = ri.to(config.device)
        mask = attention_rollout(rollout_input)
        save_rollout_img(image_list[ri_idx], mask, rollout_path + "/" + split_part + "_" + str(ri_idx) + ".png")
    return

def split_model_(model_adapter, model_part):
    with torch.no_grad():
        layers = model_adapter.get_layers()
        for layer_idx, layer_adapter in enumerate(layers):
            if model_part == "sparse":
                replaced_layers = find_layers(layer_adapter.layer, layers=[CompressedLinear])
                for layer_name in replaced_layers:
                    replaced_layers[layer_name].V.data *= 0
                    replaced_layers[layer_name].U.data *= 0
            elif model_part == "low_rank":
                replaced_layers = find_layers(layer_adapter.layer, layers=[CompressedLinear])
                for layer_name in replaced_layers:
                    replaced_layers[layer_name].S.data *= 0 

# Code based on: https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py
def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.cpu().mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.cpu().max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.cpu().min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask)) # normalize to make sure lower bounded by 0 and upper bounded by 1
    return mask    

class AttentionRollout:
    def __init__(self, model, head_fusion="mean", discard_ratio=0.4):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
    
    def __call__(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor, output_attentions=True)
        return rollout(output.attentions, self.discard_ratio, self.head_fusion)
        
def save_rollout_img(img, mask, img_name):
    def show_mask_on_image(img, mask):
        img = np.float32(img) / 255
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)
    
    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)

    cv2.imwrite(img_name, mask)