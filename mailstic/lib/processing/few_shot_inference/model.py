from transformers import AutoModel
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class Model(nn.Module):
    
    def __init__(self, model_id):
        super(Model, self).__init__()
        model_cfg = AutoConfig.from_pretrained(model_id)
        self.embedder = AutoModel.from_config(model_cfg)

    def get_embed(self, text_encoded):
        return torch.nn.functional.normalize(self.embedder(**text_encoded)[1])
        
    def forward(self, anchors, positives=None, negatives=None):
        
        anchors_embeds = self.get_embed(anchors)
        if positives is None:
            return anchors_embeds
        positives_embeds = self.get_embed(positives)
        negatives_embeds = self.get_embed(negatives)
        
        return anchors_embeds, positives_embeds, negatives_embeds