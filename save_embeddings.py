import os
import argparse
import numpy as np
from dataset import split_data
from inference import precompute_dataset_embeddings, save_embeddings


def main():
    parser = argparse.ArgumentParser(description='Precompute and save embeddings')
    parser.add_argument('--data-dir', default='caltech-101', help='Path to dataset')
    parser.add_argument('--weights-dir', default='weights', help='Directory with model checkpoints')
    parser.add_argument('--output-dir', default='embeddings', help='Directory to save embeddings')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split data
    train_data, val_data, test_data = split_data(args.data_dir)
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    experiments = [
        ('contrastive_model_best.pt', 'contrastive'),
        ('triplet_random_model_best.pt', 'triplet_random'),
        ('triplet_hard_model_best.pt', 'triplet_hard')
    ]
    
    for checkpoint_file, exp_name in experiments:
        checkpoint_path = os.path.join(args.weights_dir, checkpoint_file)
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        # Create separate folder for each experiment
        exp_output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)
    
        print(f"Computing embeddings for: {exp_name}")
  
        # Train embeddings
        print("Computing train embeddings...")
        train_emb = precompute_dataset_embeddings(
            train_paths, checkpoint_path,
            batch_size=args.batch_size
        )
        train_emb_file = os.path.join(exp_output_dir, f'embeddings_train.npy')
        save_embeddings(train_emb, train_emb_file)
        np.save(os.path.join(exp_output_dir, 'labels_train.npy'), np.array(train_labels))
        
        # Val embeddings
        print("Computing validation embeddings...")
        val_emb = precompute_dataset_embeddings(
            val_paths, checkpoint_path,
            batch_size=args.batch_size
        )
        val_emb_file = os.path.join(exp_output_dir, f'embeddings_val.npy')
        save_embeddings(val_emb, val_emb_file)
        np.save(os.path.join(exp_output_dir, 'labels_val.npy'), np.array(val_labels))
        
        # Test embeddings
        print("Computing test embeddings...")
        test_emb = precompute_dataset_embeddings(
            test_paths, checkpoint_path,
            batch_size=args.batch_size
        )
        test_emb_file = os.path.join(exp_output_dir, f'embeddings_test.npy')
        save_embeddings(test_emb, test_emb_file)
        np.save(os.path.join(exp_output_dir, 'labels_test.npy'), np.array(test_labels))
        np.save(os.path.join(exp_output_dir, 'paths_test.npy'), np.array(test_paths), allow_pickle=True)
        
        print(f"Embeddings and labels saved in: {exp_output_dir}")
      
    print("Embeddings creation job done")
    print(f"Embeddings saved to: {args.output_dir}")


if __name__ == '__main__':
    main()


