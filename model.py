import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.models as models 


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim:int=128) -> None:
        super().__init__()
        #Resnet50 with pretrained weights as backbone feature_extractor
        backbone=models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        input_dim=backbone.fc.in_features 
        #striping final fc layer from backbone 
        backbone.fc=nn.Identity()
        self.feature_extractor=backbone
        #Fully Connected Layer(2048 -> 128)
        self.fc=nn.Linear(input_dim,embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # step1: extaract features using Resnet50
        features=self.feature_extractor(x)
        #step2: prohect fetaures to 128 embedding vector 
        embeddings=self.fc(features)
        #step3: L2 normalization
        embeddings=F.normalize(embeddings) # by default L2 on rows 
        return embeddings
        

