from typing import List, Tuple, Any, Optional
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def load_dataset(data_dir: str) -> Tuple[List[str], List[int]]:
    
    image_paths: List[str] = []
    labels: List[int] = []
    
    
    valid_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    valid_folders.sort() 
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(valid_folders)}

    for label_name in valid_folders:
        cls_path = os.path.join(data_dir, label_name)
        for file in os.listdir(cls_path):
            full_path = os.path.join(cls_path, file)
            image_paths.append(full_path)
            
            
            labels.append(class_to_idx[label_name])
            
    
    return image_paths, labels


def split_data(
    data_dir: str,
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
   
    image_paths, labels = load_dataset(data_dir)
    n = len(image_paths)
    
    
    indxes = list(range(n))
    random.shuffle(indxes)
    
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    
    train_indxes = indxes[:train_end]
    val_indxes = indxes[train_end:val_end]
    test_indxes = indxes[val_end:]

    train_paths = [image_paths[i] for i in train_indxes]
    train_labels = [labels[i] for i in train_indxes]

    val_paths = [image_paths[i] for i in val_indxes]
    val_labels = [labels[i] for i in val_indxes]

    test_paths = [image_paths[i] for i in test_indxes]
    test_labels = [labels[i] for i in test_indxes]

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels), 
    )



class BaseMetricDataset(Dataset):
   
    def __init__(self, image_paths: List[str], labels: List[int], transforms: Optional[transforms.Compose]=None) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
         
        self.label_to_idx = {}
        for index, label in enumerate(labels):
            if label not in self.label_to_idx:
                self.label_to_idx[label] = []
            self.label_to_idx[label].append(index)
            
        self.classes = list(self.label_to_idx.keys())

    def __len__(self) -> int:
        return len(self.image_paths)


class ContrastiveDataset(BaseMetricDataset):
    
    def __init__(self, image_paths: List[str], labels: List[int], transforms: Optional[transforms.Compose]=None, prob: float=0.5) -> None:
        super().__init__(image_paths, labels, transforms)
        self.prob = prob

    def __getitem__(self, index: int) -> Tuple[Any, Any, torch.Tensor]:
        img1_path = self.image_paths[index]
        label1 = self.labels[index]
        
        # positive pair
        if random.random() < self.prob:
            index2 = random.choice(self.label_to_idx[label1])
            label = 1.0
        else:
            # Negative pair
            different_class = random.choice(self.classes)
            while different_class == label1:
                different_class = random.choice(self.classes)
            
            index2 = random.choice(self.label_to_idx[different_class])
            label = 0.0

        img2_path = self.image_paths[index2]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class TripletDataset(BaseMetricDataset):
    
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        anchor_path = self.image_paths[index]
        anchor_label = self.labels[index]

        # Sample Positive
        positive_indices = self.label_to_idx[anchor_label]
        pos_index = random.choice(positive_indices)
        while pos_index == index and len(positive_indices) > 1:
            pos_index = random.choice(positive_indices)
        positive_path = self.image_paths[pos_index]

        # Sample Negative
        negative_class = random.choice(self.classes)
        while negative_class == anchor_label:
            negative_class = random.choice(self.classes)
        neg_index = random.choice(self.label_to_idx[negative_class])
        negative_path = self.image_paths[neg_index]

        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transforms:
            anchor_img = self.transforms(anchor_img)
            positive_img = self.transforms(positive_img)
            negative_img = self.transforms(negative_img)

        return anchor_img, positive_img, negative_img
