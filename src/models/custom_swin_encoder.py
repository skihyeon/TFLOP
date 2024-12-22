import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.swin_transformer_v2 import SwinTransformerV2
import torch
import torch.nn as nn
from typing import List, Union, Optional
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
import math
import torch.nn.functional as F
import PIL
import numpy as np
from PIL import ImageOps

class CustomSwinEncoder(nn.Module):
    def __init__(
            self,
            input_size: List[int],
            align_long_axis: bool,
            window_size: int,
            encoder_layer: List[int]):
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer

        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        
        self.model = SwinTransformerV2(
            img_size = self.input_size,
            depths = self.encoder_layer,
            window_size = self.window_size,
            patch_size = 4,
            embed_dim = 128,
            num_heads = [4,8,16,32],
            num_classes=0
        )
        self.model.norm = None
        self.feature_dim = self.model.num_features

        swin_state_dict = timm.create_model("swinv2_base_window8_256", pretrained=True).state_dict()
        new_swin_state_dict = self.model.state_dict()

        for x in new_swin_state_dict:
            if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                pass
            elif (
                x.endswith("relative_position_bias_table")
                and self.model.layers[0].blocks[0].attn.window_size[0] != 8
            ):
                pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                old_len = int(math.sqrt(len(pos_bias)))
                new_len = int(2 * window_size - 1)
                pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(0, 3, 1, 2)
                pos_bias = F.interpolate(pos_bias, size=(new_len, new_len), mode="bicubic", align_corners=False)
                new_swin_state_dict[x] = pos_bias.permute(0, 2, 3, 1).reshape(1, new_len ** 2, -1).squeeze(0)
            else:
                new_swin_state_dict[x] = swin_state_dict[x]
    
        self.model.load_state_dict(new_swin_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.model.patch_embed.proj.weight.device)
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)
        # 각 레이어를 순차적으로 통과
        for layer in self.model.layers:
            x = layer(x)
        return x
    
    def prepare_input(self, images: List[PIL.Image.Image], random_padding: bool = False) -> torch.Tensor:
        processed_images = []
        for img in images:
            img = img.convert("RGB")
            if self.align_long_axis and (            
                (self.input_size[0] > self.input_size[1] and img.width > img.height)
                or (self.input_size[0] < self.input_size[1] and img.width < img.height)
            ):
                img = rotate()

            img = resize(img, self.input_size)
            img.thumbnail((self.input_size, self.input_size))
            delta_width = self.input_size - img.width
            delta_height = self.input_size - img.height
            if random_padding:
                pad_width = np.random.randint(low=0, high=delta_width + 1)
                pad_height = np.random.randint(low=0, high=delta_height + 1)
            else:
                pad_width = delta_width // 2
                pad_height = delta_height // 2
            padding = (
                pad_width,
                pad_height,
                delta_width - pad_width,
                delta_height - pad_height,
            )
            processed_images.append(self.to_tensor(ImageOps.expand(img, padding)))
            
        return torch.stack(processed_images)
