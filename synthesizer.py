import numpy as np

class ConvergenceLayer:
    """
    Temperature-Scaled Weighted Convergence Layer (EXPERIMENTAL / opt-in).

    Synthesizes vectors from multiple nodes dynamically by assigning learned
    independent attention weight averages (Softmaxing cross-node values)
    and scales them dynamically across positions via Temperature.

    NOTE: This layer is not yet wired into the backward pass. `backprop.py`
    assumes the nodes are merged by a plain mean, so the default `synthesize()`
    below uses an equal-weight average to keep the forward and backward passes
    mathematically consistent (and deterministic). Train this layer and route
    its gradients through `backprop.py` before using it in the live pipeline.
    """
    def __init__(self, dim, num_nodes=6):
        self.dim = dim
        self.num_nodes = num_nodes
        # Learned scalar weights per node
        self.node_weights = np.ones(num_nodes) / num_nodes
        # Position-aware linear projection
        self.U = np.random.randn(dim, 1) * np.sqrt(2.0 / dim)
        self.dW = np.zeros_like(self.node_weights)
        self.dU = np.zeros_like(self.U)
        self.cache = None

    def forward(self, node_matrices, temperature=1.0):
        # We assume node_matrices is a list of arrays (seq_len, dim)
        N = len(node_matrices)
        max_seq_len = max([mat.shape[0] for mat in node_matrices]) if node_matrices else 0
        if max_seq_len == 0:
            return np.array([])
            
        stacked = np.zeros((N, max_seq_len, self.dim))
        for i, mat in enumerate(node_matrices):
            seq_len = mat.shape[0]
            stacked[i, :seq_len, :] = mat
            
        # Calculate unnormalized scores for each node at each position
        scores = np.zeros((max_seq_len, N))
        for i in range(N):
            # stacked[i] is (max_seq_len, dim). U is (dim, 1)
            # Add learned node weight bias
            scores[:, i] = (stacked[i] @ self.U).squeeze(-1) + self.node_weights[i]
            
        # Apply temperature scaling and Softmax across nodes for each position
        scores = scores / temperature
        
        # Softmax computation
        # np.exp(scores - max) for numerical stability
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        out = np.zeros((max_seq_len, self.dim))
        for i in range(N):
            # Broadcast attention weights from (max_seq_len,) to (max_seq_len, dim)
            weight_col = attn_weights[:, i:i+1] 
            out += weight_col * stacked[i]
            
        self.cache = (stacked, scores, attn_weights, out)
        return out
        
def synthesize(nodes_outputs):
    """
    Takes the output dictionaries from multiple nodes and merges them together
    into a single sequence of vectors.

    Nodes may emit sequences of differing lengths (their system prompts differ),
    so shorter sequences are zero-padded up to the longest before averaging.
    Uses an equal-weight mean across nodes, which is deterministic and matches
    the gradient distribution in `backprop.py` (each node receives 1/N of the
    downstream gradient). See `ConvergenceLayer` for the experimental learned
    alternative.
    """
    if not nodes_outputs:
        return []

    # Extract matrices, one (seq_len, dim) array per node
    matrices = []
    for res in nodes_outputs:
        vecs = res.get("output_vectors", [])
        if len(vecs) > 0:
            matrices.append(np.array(vecs))

    if len(matrices) == 0:
        return []

    max_seq_len = max(mat.shape[0] for mat in matrices)
    dim = matrices[0].shape[1]

    # Pad every node's sequence to the same length, then average across nodes
    stacked = np.zeros((len(matrices), max_seq_len, dim))
    for i, mat in enumerate(matrices):
        stacked[i, : mat.shape[0], :] = mat

    synthesized_matrix = np.mean(stacked, axis=0)

    return synthesized_matrix.tolist()