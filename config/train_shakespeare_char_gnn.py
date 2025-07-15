from config.train_shakespeare_char import *  # baseline hyper‑params

# Reduce model size for ~1M parameters and faster CPU training
n_layer = 8        # fewer transformer layers
n_head = 8         # fewer attention heads
n_embd = 128        # smaller embedding dimension
batch_size = 16    # smaller batch for CPU
block_size = 256   # smaller context window (optional, for speed)
dropout = 0.2      # keep dropout for regularization

max_iters = 1000

n_front = 5        # transformer layers before GNN
n_graph = 3        # number of GraphBlocks
learning_rate = 1e-3  # as before
edge_recipe = "window5"  # if you parameterise recipes
device = 'cpu'
compile = False  # (Optional, disables torch.compile which is CUDA-optimized)