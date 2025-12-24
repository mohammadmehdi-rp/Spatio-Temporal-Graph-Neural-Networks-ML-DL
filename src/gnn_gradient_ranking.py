import numpy as np
import torch
from torch.utils.data import DataLoader
from your_dataset_module import NowcastDataset  # whatever you use in train_nowcast_sparse
from your_model_module import YourGNNModel     # same model as in train_nowcast_sparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_node_importance(model, val_loader, n_nodes, save_debug=False):
    model.eval()
    node_scores = torch.zeros(n_nodes, dtype=torch.float32, device=DEVICE)
    total_batches = 0

    for batch in val_loader:
        # adapt these lines to your actual batch structure
        # Example assumption:
        #   batch_X: [B, T, N, F]
        #   batch_y: [B, T, N] or similar
        batch_X = batch["X"].to(DEVICE)
        batch_y = batch["Y"].to(DEVICE)

        # we need gradients w.r.t. X
        batch_X.requires_grad_(True)

        # forward + loss
        y_pred = model(batch_X)          # adapt to your signature
        loss = torch.nn.functional.mse_loss(y_pred, batch_y)

        # backward
        model.zero_grad()
        if batch_X.grad is not None:
            batch_X.grad.zero_()
        loss.backward()

        # batch_X.grad: [B, T, N, F]
        grad = batch_X.grad.detach()
        B, T, N, F = grad.shape

        # compute per-node L2 norm over features, then mean over time & batch
        # grad_norm_per_node: [N]
        grad_norm = grad.pow(2).sum(dim=3).sqrt()   # [B, T, N]
        grad_norm_mean = grad_norm.mean(dim=(0,1))  # [N]

        node_scores += grad_norm_mean
        total_batches += 1

    node_scores /= max(total_batches, 1)
    return node_scores

def main():
    # 1) load dataset & build val loader
    npz_path = "dataset_sparse_10.npz"
    Z = np.load(npz_path, allow_pickle=True)
    nodes = Z["nodes"]
    n_nodes = nodes.shape[0]

    # Use the same Dataset/DataLoader as in train_nowcast_sparse.py
    val_dataset = NowcastDataset(npz_path, split="val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 2) load model
    model = YourGNNModel(...)  # same args as in train_nowcast_sparse
    model.load_state_dict(torch.load("nowcast_10.pt", map_location=DEVICE))
    model.to(DEVICE)

    # 3) compute importance
    node_scores = compute_node_importance(model, val_loader, n_nodes)
    node_scores_cpu = node_scores.detach().cpu().numpy()

    # 4) sort & print ranking
    order = np.argsort(-node_scores_cpu)  # descending
    print("=== GNN gradient-based node ranking ===")
    for rank, idx in enumerate(order):
        print(f"{rank+1:2d}. {nodes[idx]}  score={node_scores_cpu[idx]:.6f}")

    # optionally save to file for later processing
    np.savez("gnn_node_importance_10.npz",
             nodes=nodes,
             scores=node_scores_cpu,
             order=order)

if __name__ == "__main__":
    main()
