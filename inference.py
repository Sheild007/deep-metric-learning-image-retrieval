import torch
import numpy as np
from typing import Union, List
from PIL import Image
import os
from model import EmbeddingNet
from train import get_val_transforms

def load_model(checkpoint_path: str, device: str = None) -> tuple[EmbeddingNet, str]:
    if device is None:
       if torch.cuda.is_available():
             device = 'cuda'  
       else:
            device = 'cpu'
    
    model = EmbeddingNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device

def generate_embeddings(image_paths: Union[str, List[str]], checkpoint_path: str) -> np.ndarray:
    model, device = load_model(checkpoint_path) #lodding model
    transform = get_val_transforms()# applying tranforms
    if isinstance(image_paths, str):
        image_paths = [image_paths] # converting single img to list for consistency
    
    embeddings = []
    with torch.no_grad():
        for image_path in image_paths:
            img = Image.open(image_path).convert('RGB')#loafding image
            img_tensor = transform(img).unsqueeze(0).to(device)# adding batch layer for consistency and loading on device
            embedding = model(img_tensor)#computing embeding
            embeddings.append(embedding.cpu().numpy())# off loadinf to cpu and converting to numpy
    
    return np.vstack(embeddings)

def precompute_dataset_embeddings(image_paths: List[str], checkpoint_path: str, batch_size: int = 32) -> np.ndarray:
    model, device = load_model(checkpoint_path)
    transform = get_val_transforms()
    
    embeddings = []    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            batch_images.append(img_tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_embeddings = model(batch_tensor)
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)


def save_embeddings(embeddings: np.ndarray, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path) or 'embeddings', exist_ok=True)
    np.save(save_path, embeddings)
    print(f"Embeddings saved to {save_path}")


def load_embeddings(save_path: str) -> np.ndarray:
    embeddings = np.load(save_path)
    print(f"Embeddings loaded from {save_path}")
    return embeddings
