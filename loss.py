import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        #D = ||f(x1) - f(x2)||_2
        dist=F.pairwise_distance(output1,output2)
        #Loss = y * D^2 + (1 - y) * max(0, margin - D)^2
        loss = torch.mean(
            label * torch.pow(dist, 2) +
            (1.0 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # D(a, p) + margin < D(a, n)
        dist1=F.pairwise_distance(anchor,positive)
        dist2=F.pairwise_distance(anchor,negative)
        #Loss = max(0, D(anchor, positive) - D(anchor, negative) + margin)
        loss=torch.mean(torch.clamp(dist1-dist2+self.margin,min=0.0))
        return loss

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # This computes the distance between every single image in the batch
        distances=torch.cdist(embeddings,embeddings)
        is_same_label=labels.unsqueeze(0)==labels.unsqueeze(1) #bolean matrix representing same clas
        # hard positive will be max dist in sasme class
        all_same_class_dist=distances*is_same_label.float()
        hard_positive,_=all_same_class_dist.max(dim=1) # taking max across rows 
        
        # hard negative will be min example in all different classes
        max_dist=distances.max()
        all_diff_class_dist=distances+(is_same_label.float()*max_dist)# assigning same class max dist so that they are not slelected
        hard_neg,_=all_diff_class_dist.min(dim=1)
        #Loss = max(0, D(anchor, positive) - D(anchor, negative) + margin)
        loss=torch.mean(torch.clamp(hard_positive- hard_neg+self.margin,min=0.0))
        return loss
        
        
