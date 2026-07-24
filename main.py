import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import train_contrastive, train_triplet_random, train_triplet_hard
from inference import precompute_dataset_embeddings, save_embeddings
from dataset import split_data
from retrieval import recall_at_k, plot_tsne, show_retrieval


def train_all_models(args: argparse.Namespace, config: dict) -> dict:
    print("Training models...")
    checkpoints = {}
    print("Training: Contrastive Loss + Random Pairs")
    checkpoint1 = train_contrastive(
        data_dir=args.data_dir, output_dir=args.checkpoint_dir, margin=1.0, **config
    )
    checkpoints["contrastive"] = checkpoint1

    print("Training: Triplet Loss + Random Triplets")
    checkpoint2 = train_triplet_random(
        data_dir=args.data_dir, output_dir=args.checkpoint_dir, margin=0.2, **config
    )
    checkpoints["triplet_random"] = checkpoint2

    print("Training: Triplet Loss + Hard Negative Mining")
    checkpoint3 = train_triplet_hard(
        data_dir=args.data_dir, output_dir=args.checkpoint_dir, margin=0.2, **config
    )
    checkpoints["triplet_hard"] = checkpoint3

    return checkpoints


def generate_and_save_embeddings(args: argparse.Namespace) -> dict:
    print("Generating and saving embeddings...")

    os.makedirs(args.embeddings_dir, exist_ok=True)

    # Split data
    train_data, val_data, test_data = split_data(args.data_dir)
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data

    experiments = [
        ("contrastive_model_best.pt", "contrastive"),
        ("triplet_random_model_best.pt", "triplet_random"),
        ("triplet_hard_model_best.pt", "triplet_hard"),
    ]

    embeddings_data = {}

    for checkpoint_file, exp_name in experiments:
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_file)

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue

        exp_output_dir = os.path.join(args.embeddings_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)

        print(f"Computing embeddings for: {exp_name}")
        print("Computing train embeddings...")
        train_emb = precompute_dataset_embeddings(
            train_paths, checkpoint_path, batch_size=args.batch_size
        )

        print("Computing validation embeddings...")
        val_emb = precompute_dataset_embeddings(
            val_paths, checkpoint_path, batch_size=args.batch_size
        )

        print("Computing test embeddings...")
        test_emb = precompute_dataset_embeddings(
            test_paths, checkpoint_path, batch_size=args.batch_size
        )

        print("Saving embeddings...")
        save_embeddings(train_emb, os.path.join(exp_output_dir, "embeddings_train.npy"))
        np.save(os.path.join(exp_output_dir, "labels_train.npy"), train_labels)
        np.save(os.path.join(exp_output_dir, "paths_train.npy"), train_paths)

        save_embeddings(val_emb, os.path.join(exp_output_dir, "embeddings_val.npy"))
        np.save(os.path.join(exp_output_dir, "labels_val.npy"), val_labels)
        np.save(os.path.join(exp_output_dir, "paths_val.npy"), val_paths)

        save_embeddings(test_emb, os.path.join(exp_output_dir, "embeddings_test.npy"))
        np.save(os.path.join(exp_output_dir, "labels_test.npy"), test_labels)
        np.save(os.path.join(exp_output_dir, "paths_test.npy"), test_paths)

        embeddings_data[exp_name] = {
            "test_emb": test_emb,
            "test_labels": test_labels,
            "test_paths": test_paths,
        }

        print(f"Embeddings saved to {exp_output_dir}")

    return embeddings_data


def evaluate_embeddings(embeddings_data: dict, args: argparse.Namespace) -> dict:
    print("Evaluating embeddings...")

    os.makedirs(args.graphs_dir, exist_ok=True)
    results = {}

    for exp_name, data in embeddings_data.items():
        print(f"Evaluating: {exp_name}")

        test_emb = data["test_emb"]
        test_labels = data["test_labels"]
        test_paths = data["test_paths"]

        exp_output_dir = os.path.join(args.graphs_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)

        print("Computing Recall@K metrics...")
        r1 = recall_at_k(test_emb, test_labels, k=1)
        r5 = recall_at_k(test_emb, test_labels, k=5)
        r10 = recall_at_k(test_emb, test_labels, k=10)

        print(f"Recall@1: {r1:.4f}")
        print(f"Recall@5: {r5:.4f}")
        print(f"Recall@10: {r10:.4f}")

        results[exp_name] = {"recall@1": r1, "recall@5": r5, "recall@10": r10}

        print("Generating t-SNE visualization...")
        tsne_output = os.path.join(exp_output_dir, "tsne.png")
        plot_tsne(
            test_emb,
            test_labels,
            title=f"{exp_name} - t-SNE Visualization",
            output_path=tsne_output,
        )

        print("Generating retrieval visualizations...")
        retrieval_dir = os.path.join(exp_output_dir, "retrieval")
        os.makedirs(retrieval_dir, exist_ok=True)

        n_queries = min(10, len(test_paths))
        query_indices = np.linspace(0, len(test_paths) - 1, n_queries, dtype=int)

        for i, query_idx in enumerate(query_indices):
            output_path = os.path.join(retrieval_dir, f"query_{i:02d}.png")
            show_retrieval(
                int(query_idx),
                test_emb,
                test_paths,
                test_labels,
                k=5,
                output_path=output_path,
            )
            print(f"Query {i+1}/{n_queries} saved")

        print(f"Evaluation complete for {exp_name}")
        print(f"Results saved to: {exp_output_dir}")

    return results


def print_summary(results: dict) -> None:
    print("\nFinal Summary - Recall@K Metrics:")
    print(f"{'Model':<20} {'Recall@1':<15} {'Recall@5':<15} {'Recall@10':<15}")

    for model_name, metrics in results.items():
        print(
            f"{model_name:<20} {metrics['recall@1']:<15.4f} {metrics['recall@5']:<15.4f} {metrics['recall@10']:<15.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Complete Embedding Learning Pipeline")

    parser.add_argument("--data-dir", default="caltech-101", help="Path to dataset")
    parser.add_argument(
        "--checkpoint-dir",
        default="weights",
        help="Directory for saving model checkpoints",
    )
    parser.add_argument(
        "--embeddings-dir", default="embeddings", help="Directory for saving embeddings"
    )
    parser.add_argument(
        "--graphs-dir", default="graphs", help="Directory for saving visualizations"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and inference",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip training phase")
    parser.add_argument(
        "--skip-embeddings", action="store_true", help="Skip embedding generation phase"
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip evaluation phase"
    )

    args = parser.parse_args()

    print("Starting embedding learning pipeline...")

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.embeddings_dir, exist_ok=True)
    os.makedirs(args.graphs_dir, exist_ok=True)

    embeddings_data = {}
    results = {}

    if not args.skip_train:
        train_all_models(args, config)
    else:
        print("Skipping training phase (--skip-train)")

    if not args.skip_embeddings:
        embeddings_data = generate_and_save_embeddings(args)
    else:
        print("Skipping embedding generation (--skip-embeddings)")
        # Load existing embeddings if available
        for exp_name in ["contrastive", "triplet_random", "triplet_hard"]:
            exp_dir = os.path.join(args.embeddings_dir, exp_name)
            if os.path.exists(exp_dir):
                try:
                    test_emb = np.load(os.path.join(exp_dir, "embeddings_test.npy"))
                    test_labels = np.load(os.path.join(exp_dir, "labels_test.npy"))
                    test_paths = np.load(
                        os.path.join(exp_dir, "paths_test.npy"), allow_pickle=True
                    )

                    # Fix paths: extract class/image from old paths and reconstruct with local dir
                    test_paths_fixed = []
                    for p in test_paths:
                        if isinstance(p, (str, np.str_)):
                            p_str = str(p)
                            # Extract the last two parts: class_name/image_name.jpg
                            parts = p_str.split("/")
                            if len(parts) >= 2:
                                class_name = parts[-2]
                                image_name = parts[-1]
                                p_str = os.path.join(
                                    args.data_dir, class_name, image_name
                                )
                            test_paths_fixed.append(p_str)
                        else:
                            test_paths_fixed.append(p)
                    test_paths = np.array(test_paths_fixed, dtype=object)

                    embeddings_data[exp_name] = {
                        "test_emb": test_emb,
                        "test_labels": test_labels,
                        "test_paths": test_paths,
                    }
                except Exception as e:
                    print(f"Could not load embeddings for {exp_name}: {e}")

    if not args.skip_eval and embeddings_data:
        results = evaluate_embeddings(embeddings_data, args)
        print_summary(results)
    elif args.skip_eval:
        print("Skipping evaluation phase (--skip-eval)")
    else:
        print("No embeddings available for evaluation")

    print("\nPipeline complete.")
    print(f"Results saved to: {args.graphs_dir}")


if __name__ == "__main__":
    main()
