import torch
import torch.nn as nn
import torch.nn.functional as F

class LayoutPointer(nn.Module):
    def __init__(self, feature_dim, temperature):
        super().__init__()
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.empty_proj = nn.Linear(feature_dim, feature_dim)
        self.box_proj = nn.Linear(feature_dim, feature_dim)
        self.tag_proj = nn.Linear(feature_dim, feature_dim)
        self.temperature = temperature

    def forward(self, decoder_hidden_states, layout_prompt_length):
        B = decoder_hidden_states.size(0)
        box_features = decoder_hidden_states[:, :layout_prompt_length]
        tag_features = decoder_hidden_states[:, layout_prompt_length:]
        
        box_features = F.normalize(self.box_proj(box_features), dim=-1)
        tag_features = F.normalize(self.tag_proj(tag_features), dim=-1)
        
        empty_embedding = self.empty_embedding.expand(B, -1, -1)
        empty_features = F.normalize(self.empty_proj(empty_embedding), dim=-1)
        
        pointer_logits = torch.matmul(box_features, tag_features.transpose(-2, -1))
        empty_pointer_logits = torch.matmul(empty_features, tag_features.transpose(-2, -1))
        
        return pointer_logits, empty_pointer_logits