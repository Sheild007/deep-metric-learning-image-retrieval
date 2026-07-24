import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import EmbeddingNet
from dataset import split_data, ContrastiveDataset, TripletDataset
from loss import ContrastiveLoss, TripletLoss, BatchHardTripletLoss

GLOBAL_BATCH_SIZE = 64
GLOBAL_LR = 0.001
GLOBAL_WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

def get_transforms() -> transforms.Compose:
    # return data augmentation and normalization transforms
    return transforms.Compose([
        transforms.Resize((224, 224)), #resnet50 accept these dims
        transforms.RandomHorizontalFlip(p=0.5), # for augmentation
        transforms.RandomRotation(10),          #  for augmentation
        transforms.ToTensor(),                  # conversion to tensor
        transforms.Normalize(                   # offical normalize mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms() -> transforms.Compose:
    #return normalization transforms
    return transforms.Compose([          
        transforms.Resize((224, 224)),  #resnet50 accept these dims
        transforms.ToTensor(),          # conversion to tensor
        transforms.Normalize(           # offical normalize mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def save_loss_plot(train_losses: list, val_losses: list, epochs: int, output_path: str = 'graphs/loss_plot.png', title: str = 'Training and Validation Loss', xlabel: str = 'Epoch', ylabel: str = 'Loss') -> None:
    """Save training and validation loss plots to file."""
    os.makedirs(os.path.dirname(output_path) or 'graphs', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', marker='o', linewidth=2)
    ax.plot(range(1, len(train_losses)+1), val_losses, label='Validation Loss', marker='s', linewidth=2)
    ax.set_xticks(range(1, len(train_losses) + 1))
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()

def train_contrastive(data_dir: str = 'caltech-101', output_dir: str = 'weights',
    epochs: int = 30, batch_size: int = GLOBAL_BATCH_SIZE, lr: float = GLOBAL_LR, weight_decay: float = GLOBAL_WEIGHT_DECAY, margin: float = 1.0, device: str = None) -> str:
   
    if device is None:
       if torch.cuda.is_available():
             device = 'cuda'  
       else:
            device = 'cpu' 
    
    #spliting data with default 70 15 15 ratio
    train_data, val_data, _ = split_data(data_dir)
    
    train_transforms=get_transforms()
    val_transforms=get_val_transforms()
    
    #making dataset and data loaders
    train_dataset=ContrastiveDataset(train_data[0],train_data[1],transforms=train_transforms,prob=0.5)
    val_dataset=ContrastiveDataset(val_data[0],val_data[1],transforms=val_transforms,prob=0.5)
    
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    
    #loading the model
    model=EmbeddingNet().to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    criterion=ContrastiveLoss(margin=margin)
    
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = os.path.join(output_dir, 'contrastive_model_best.pt')
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print(f"Contrastive Learning: Training for {epochs} epochs (Early Stopping Patience: {EARLY_STOPPING_PATIENCE})...")
    for epoch in range(epochs):
        model.train() # swtich model to train phase
        train_loss = 0.0
        for img1,img2,label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img1=img1.to(device)
            img2=img2.to(device)
            label=label.to(device)
            
            optimizer.zero_grad() # zeroing gradients
            emb1=model(img1)      # embeddings of img1
            emb2=model(img2)      # embeddings of img2

            loss=criterion(emb1, emb2, label)  # computing contrastive loss
            
            loss.backward() # back prop
            optimizer.step() # updating weights
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval() # switch back to eval mode
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                label = label.to(device)
                
                emb1 = model(img1)
                emb2 = model(img2)
                loss = criterion(emb1, emb2, label)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"Best model saved with val_loss={val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
                break

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    save_loss_plot(train_losses, val_losses, len(train_losses), output_path='graphs/contrastive/loss_plot.png')
    
    return checkpoint

def train_triplet_random(data_dir: str = 'caltech-101', output_dir: str = 'weights',
    epochs: int = 30, batch_size: int = GLOBAL_BATCH_SIZE, lr: float = GLOBAL_LR, weight_decay: float = GLOBAL_WEIGHT_DECAY, margin: float = 0.2, device: str = None) -> str:
    
    if device is None:
       if torch.cuda.is_available():
             device = 'cuda'  
       else:
            device = 'cpu' 
    
    
    train_data, val_data, _ = split_data(data_dir)
    
    train_transforms = get_transforms()
    val_transforms = get_val_transforms()
    
    train_dataset = TripletDataset(train_data[0], train_data[1], transforms=train_transforms)
    val_dataset = TripletDataset(val_data[0], val_data[1], transforms=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = EmbeddingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = TripletLoss(margin=margin)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('graphs', exist_ok=True)
    checkpoint = os.path.join(output_dir, 'triplet_random_model_best.pt')
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print(f"Triplet Loss (Random): Training for {epochs} epochs (Early Stopping Patience: {EARLY_STOPPING_PATIENCE})...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            emb_anchor = model(anchor)
            emb_pos = model(positive)
            emb_neg = model(negative)
            loss = criterion(emb_anchor, emb_pos, emb_neg)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                emb_anchor = model(anchor)
                emb_pos = model(positive)
                emb_neg = model(negative)
                loss = criterion(emb_anchor, emb_pos, emb_neg)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"Best model saved with val_loss={val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
                break
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    save_loss_plot(train_losses, val_losses, len(train_losses), output_path='graphs/triplet_random/loss_plot.png')
    
    return checkpoint

def train_triplet_hard(data_dir: str = 'caltech-101', output_dir: str = 'weights',
    epochs: int = 30, batch_size: int = GLOBAL_BATCH_SIZE, lr: float = GLOBAL_LR, weight_decay: float = GLOBAL_WEIGHT_DECAY, margin: float = 0.2, device: str = None) -> str:
    
    if device is None:
       if torch.cuda.is_available():
             device = 'cuda'  
       else:
            device = 'cpu' 
    
    train_data, val_data, _ = split_data(data_dir)
    
    train_transforms = get_transforms()
    val_transforms = get_val_transforms()
    
    train_dataset = TripletDataset(train_data[0], train_data[1], transforms=train_transforms)
    val_dataset = TripletDataset(val_data[0], val_data[1], transforms=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = EmbeddingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = BatchHardTripletLoss(margin=margin)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('graphs', exist_ok=True)
    checkpoint = os.path.join(output_dir, 'triplet_hard_model_best.pt')
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print(f"Triplet Loss (Hard Mining): Training for {epochs} epochs (Early Stopping Patience: {EARLY_STOPPING_PATIENCE})...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Stack embeddings for batch hard triplet loss
            embeddings = torch.cat([anchor, positive, negative], dim=0)
            labels_anchor = torch.arange(anchor.size(0), device=device)
            labels_pos = torch.arange(anchor.size(0), device=device)
            labels_neg = torch.arange(anchor.size(0), device=device)
            labels = torch.cat([labels_anchor, labels_pos, labels_neg], dim=0)
            
            embeddings_pred = model(embeddings)
            
            optimizer.zero_grad()
            loss = criterion(embeddings_pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                embeddings = torch.cat([anchor, positive, negative], dim=0)
                labels_anchor = torch.arange(anchor.size(0), device=device)
                labels_pos = torch.arange(anchor.size(0), device=device)
                labels_neg = torch.arange(anchor.size(0), device=device)
                labels = torch.cat([labels_anchor, labels_pos, labels_neg], dim=0)
                
                embeddings_pred = model(embeddings)
                loss = criterion(embeddings_pred, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"Best model saved with val_loss={val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (patience={EARLY_STOPPING_PATIENCE})")
                break
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    save_loss_plot(train_losses, val_losses, len(train_losses), output_path='graphs/triplet_hard/loss_plot.png')
    
    return checkpoint




