import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

def recall_at_k(embeddings:np.ndarray,labels:np.ndarray,k:int=1)->float:
    
    # fining k neibioours of all embeddings
    neighbours=NearestNeighbors(n_neighbors=k + 1,algorithm='brute',metric='euclidean')
    neighbours.fit(embeddings)
    distances, indices=neighbours.kneighbors(embeddings)
    
    # Exclude query itself 
    indices=indices[:, 1:k + 1]
    retrieved_labels=labels[indices]
    query_labels=labels.reshape(-1, 1)

    # Check if correct class appears in top-k
    matches=(retrieved_labels==query_labels).any(axis=1)
    # taking avearage of all correct and wrong answers
    return np.mean(matches)

def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, title: str = 't-SNE', output_path: str = 'graphs/tsne.png') -> None:
   
    print(f"Computing t-SNE (num_comp:2")
    # taking default numcopm=2 and perplexity value 30
    tsne = TSNE(random_state=42, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create combined colormap for 101 classes (3 × 20 = 60+ unique colors)
    colors = (list(cm.get_cmap('tab20').colors) + 
              list(cm.get_cmap('tab20b').colors) + 
              list(cm.get_cmap('tab20c').colors))
    custom_cmap = ListedColormap(colors)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0],embeddings_2d[:, 1],c=labels,cmap=custom_cmap,s=30,alpha=0.7)
    plt.colorbar(scatter, label='Class (0-100)')
    plt.title(title, fontsize=14)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved to {output_path}")


def show_retrieval(query_idx:int ,embeddings:np.ndarray,image_paths:list,labels:np.ndarray,k:int=5,output_path:str=None)->None:
    neighbours = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', metric='euclidean')
    neighbours.fit(embeddings)
    query_label = labels[query_idx]
    distances, indices = neighbours.kneighbors(embeddings[query_idx].reshape(1, -1))
    # Exclude query itself
    indices = indices[0, 1:k + 1]
    distances = distances[0, 1:k + 1]
    query_img = Image.open(image_paths[query_idx]).convert('RGB').resize((224, 224))
    
    # Create grid: query + top-k neighbors
    fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 4))
    
    axes[0].imshow(query_img)
    axes[0].set_title(f'Query\nLabel: {query_label}', fontweight='bold', fontsize=11)
    axes[0].axis('off')
    
    for i, (neighbor_idx, distance) in enumerate(zip(indices, distances)):
        neighbor_img=Image.open(image_paths[neighbor_idx]).convert('RGB').resize((224, 224))
        neighbor_label=labels[neighbor_idx]
        
        axes[i + 1].imshow(neighbor_img)
        
        is_correct=(neighbor_label == query_label)
        color='green' if is_correct else 'red'
        status='Match' if is_correct else 'No Match'
        
        axes[i + 1].set_title(
            f'Top-{i+1}\n{status} | Label: {neighbor_label}\nDist: {distance:.3f}',
            color=color, fontweight='bold', fontsize=10
        )
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or 'graphs', exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Retrieval visualization saved to {output_path}")
