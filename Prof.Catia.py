import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm

np.set_printoptions(precision=4, suppress=True)
torch.manual_seed(42)

DNA = ["A", "C", "G", "T"]

MAP = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "N": [1.0, 1.0, 1.0, 1.0],
}

# ==========================================================
# HELPER FUNCTIONS & GLOBAL CLASSES
# ==========================================================
def seq2array(seq):
    return np.array([MAP[x] for x in seq], dtype=float)

def jc_Q():
    Q = np.ones((4, 4)) / 3.0
    np.fill_diagonal(Q, -1.0)
    return Q

def felsenstein_merge_step(F_left, F_right, P_left, P_right):
    n_sites = F_left.shape[0]
    F_parent = np.zeros((n_sites, 4))

    for site in range(n_sites):
        for parent_state in range(4):
            left_prob = np.sum(P_left[parent_state, :] * F_left[site, :])
            right_prob = np.sum(P_right[parent_state, :] * F_right[site, :])
            F_parent[site, parent_state] = left_prob * right_prob
    return F_parent

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def explain_tensor(name, tensor, meaning, role):
    print(f"\n{name}")
    print("Meaning:", meaning)
    print("Shape:", tensor.shape if hasattr(tensor, 'shape') else "Scalar/Primitive")
    print("Role:", role)
    print(tensor)

class SAMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, with_bias=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=with_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=with_bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class EdgeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 2500))
    def forward(self, x):
        return self.fc(x)

# ==========================================================
# GLOBAL DISCLAIMER & EXECUTION PROTOCOL
# ==========================================================
section("GLOBAL DISCLAIMER & EXECUTION PROTOCOL")
print("1. TRACE EXECUTION: This script executes the exact mathematical operations and PyTorch")
print("   architectures defined in the PhyloGFN paper, but with untuned/random parameters.")
print("2. NO CHECKPOINT: We do NOT load a pre-trained model.")
print("3. EXPLICIT MARKING: Any scalar value that relies on stochastic sampling from the untuned")
print("   network (like branch lengths) is explicitly marked as [SIMULATED] inline.")
print("4. SELF-CONTAINED: Every tensor printed corresponds directly to a mathematical concept.")

# ==========================================================
# STEP 0 & 1 — INPUT SEQUENCES & ENCODING
# ==========================================================
S = {
    "S1": "AATG",
    "S2": "AATT",
    "S3": "ATGG",
    "S4": "ACGC",
}

section("STEP 0 & 1 — INITIAL STATE (ST0) & ONE-HOT ENCODING")
print("ST0 = Initial State. ST0 contains the four input sequences: S1, S2, S3, S4.")
print("Note: S1-S4 represent Biological Sequences. ST0 represents the MDP State.\n")

encoded = {}
for name, seq in S.items():
    encoded[name] = seq2array(seq)

explain_tensor(
    "F_S1 (Example of encoded leaf)",
    encoded["S1"],
    "One-hot encoded matrix for sequence S1.",
    "Serves as the initial leaf likelihood input for Felsenstein calculations. Rows=Sites, Cols=DNA states."
)

print("\n" + "-" * 80)
print("THEORETICAL CONTEXT: DNA vs MLST")
print("-" * 80)
print("""In standard PhyloGFN, the initial feature for leaf node 'u' at site 'i' is f_u^i ∈ [0,1]^4.
For MLST (Multi-Locus Sequence Typing), the alphabet shifts to Allele IDs (1 to K). 
The vocabulary size expands from 4 to K, changing the state space from m×4 to m×K. 
For cgMLST (Core Genome), each locus requires its own transition rate matrix (Q_i) and 
mutation scaler, handled mathematically as: P_i(t) = exp(Q_i * μ_i * t).""")

# ==========================================================
# EXPLANATION — ACTION SPACE & TREE CONSTRUCTION
# ==========================================================
section("EXPLANATION — ACTION SPACE & TREE CONSTRUCTION")
print("""
1. Action Space Definition
In PhyloGFN, the state space consists of a set of disjoint rooted trees. At any non-terminal 
state containing l > 1 trees, a transition action is strictly composed of two distinct parts:
- Tree Action (Topological Merge): Selects which two disjoint subtrees to merge into a new 
  common parent node. Mathematically, the model selects a pair of trees from the C(l, 2) 
  available combinations. As the tree is built, l decreases by 1 at each step, shrinking 
  available actions from C(n, 2) down to exactly 1 candidate action.
- Edge Action (Branch Length Assignment): Once subtrees are selected, an Edge MLP jointly 
  samples a pair of evolutionary branch lengths from a discrete multinomial distribution 
  (binned values) or a continuous Gaussian mixture model.

2. Tree Construction Process
The model constructs trees using a sequential, bottom-up Markov Decision Process (MDP).
- Initial State (ST0): n completely disconnected, rooted trees (one leaf node each).
- Iterative Merging: The tree_action joins two roots, and edge_action assigns branch lengths. 
  This reduces the disconnected trees by exactly 1.
- Termination: After exactly n-1 steps, a single rooted tree remains. Because Jukes-Cantor 
  is time-reversible, the final root node is dropped to yield the unrooted bifurcating tree.
""")

# ==========================================================
# STEP 2 — DEFINING THE ACTION SPACE
# ==========================================================
section("STEP 2 — THE TOPOLOGICAL ACTION SPACE")
ST0 = list(S.keys())
actions = list(itertools.combinations(ST0, 2))

print(f"Number of available trees in ST0: {len(ST0)}")
print(f"For ST0, C(4,2) = {len(actions)} candidate merge actions:\n")

for i, (a, b) in enumerate(actions, start=1):
    print(f"  a{i}: merge({a}, {b})")

# ==========================================================
# STEP 3 — NEURAL EMBEDDING & CANONICAL G_ij VECTORS
# ==========================================================
section("STEP 3 — GENERATING CANONICAL G_ij VECTORS (Forward Policy Input)")

embedding_size = 128
input_size = 16 

seq_emb = nn.Linear(input_size, embedding_size, bias=False)
encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=4, batch_first=True)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

tree_embeddings_list = []
for name in ST0:
    y_i = torch.tensor(encoded[name].flatten(), dtype=torch.float32)
    tree_embeddings_list.append(seq_emb(y_i))

encoded_trees = torch.stack(tree_embeddings_list).unsqueeze(0) 
h_s = torch.randn(1, 1, embedding_size) 
transformer_input = torch.cat([h_s, encoded_trees], dim=1) 

with torch.no_grad():
    transformer_output = transformer(transformer_input)

summary_token = transformer_output[0, 0, :]   
e_dict = {
    "S1": transformer_output[0, 1, :],
    "S2": transformer_output[0, 2, :],
    "S3": transformer_output[0, 3, :],
    "S4": transformer_output[0, 4, :]
}

g_tensors = []
for (a, b) in actions:
    combined_tree_rep = e_dict[a] + e_dict[b]
    g_ij = torch.cat([summary_token, combined_tree_rep], dim=0) 
    g_tensors.append(g_ij)

g_batch = torch.stack(g_tensors)

explain_tensor(
    "G_S1S2 (Action 1 Canonical Vector)",
    g_batch[0][:5].tolist() + ["..."] + g_batch[0][128:133].tolist() + ["..."],
    "The 256-D feature vector representing the decision to merge S1 and S2.",
    "Formula: G_ij = [e_s ; e_i + e_j]. Indices 0-127 hold Global token. Indices 128-255 hold Local sum."
)

# ==========================================================
# EXPLANATION — FORWARD POLICY & THE "WHY"
# ==========================================================
section("EXPLANATION — FORWARD POLICY & THE 'WHY'")
print("""
3. Forward Policy vs. Random Actions
- The Learned Forward Policy (P_F): The generator evaluates the state representations (G_ij) 
  of all candidate merges and outputs raw scores (logits). Softmax converts these into a 
  valid probability distribution. Instead of greedily picking the highest score, the generator 
  stochastically samples an action according to this learned distribution.
- Random Actions (ε-greedy): During training, PhyloGFN occasionally takes completely random 
  actions with a probability ε. This forces exploration of undiscovered regions of the tree 
  space to prevent mode collapse.

4. The "Why" of the Model
The fundamental objective of the GFlowNet generator is to act as an amortized posterior sampler. 
Because the space of possible trees is super-exponential ((2n-5)!!), exhaustive search is 
impossible. The GFlowNet learns to construct trees such that the probability of generating 
a specific tree exactly mirrors its true posterior probability P(z,b | Y) ∝ R(z,b). 
By learning to sample proportionally to the reward landscape, it successfully explores suboptimal 
trees—which is biologically critical to calculate branch support and confidence intervals.
""")

# ==========================================================
# STEP 4 — LOGITS, SOFTMAX, AND GFLOWNET SAMPLING
# ==========================================================
section("STEP 4 — ACTION SCORING & GFLOWNET SAMPLING")

part1_logits_head = SAMlp(in_features=256, hidden_features=256, out_features=1)

with torch.no_grad():
    logits = part1_logits_head(g_batch).squeeze(-1)
    probabilities = F.softmax(logits, dim=0)

print("\n[Action -> Forward Probability Mapping P_F(a | ST0)]")
for idx, (a, b) in enumerate(actions):
    print(f" P_F(merge({a}, {b})) = {probabilities[idx].item() * 100:.2f}%  (Logit: {logits[idx].item():.4f})")

sampled_action_idx = torch.multinomial(probabilities, num_samples=1).item()
chosen_pair = actions[sampled_action_idx]

print("\n" + "=" * 80)
print(f"SAMPLED ACTION: merge({chosen_pair[0]}, {chosen_pair[1]})")
print("=" * 80)
print(f"State transition: ST0 --> ST1")
print(f"  - The environment merges {chosen_pair[0]} and {chosen_pair[1]}.")
print(f"  - New internal ancestral node H1 replaces them in ST1.")

# ==========================================================
# EXPLANATION — EVOLUTIONARY MODELS & FELSENSTEIN
# ==========================================================
section("EXPLANATION — EVOLUTIONARY MODELS & FELSENSTEIN PRUNING")

print("""
1. Jukes-Cantor (JC) Model
The instantaneous rate of mutation is defined by a 4×4 rate matrix Q. The diagonal elements 
(no mutation) are -1, and the off-diagonal elements are 1/3. The change in transition 
probability over time t is governed by dP(t)/dt = Q P(t).

2. Transition Matrices & Branch Lengths
Branch length t represents evolutionary distance. The model computes the matrix exponential: 
P(t) = e^(Qt). In PyTorch, this is optimized via spectral decomposition: e^(Qt) = U e^(Dt) U^-1.

3. Exponential Prior on Branch Lengths
PhyloGFN places an exponential prior (λ=10) over branch lengths: P(t) = λ * e^(-λt). 
Biologically, this aggressively penalizes unrealistically long branches to prevent saturation.

4. Felsenstein Pruning Algorithm
Equation 1 computes the partial conditional probability:
P(L_u^i | a_u^i) = Σ_{a_v^i, a_w^i ∈ Σ} [ P(a_v^i | a_u^i, b(e_v)) P(L_v^i | a_v^i) × P(a_w^i | a_u^i, b(e_w)) P(L_w^i | a_w^i) ].
It is "partial" because it is strictly conditional on the assumption that node u holds a specific state.
""")

# ==========================================================
# QUESTION 2A — EDGE MLP & FELSENSTEIN MATRICES
# ==========================================================
section("QUESTION 2A — SAMPLING BRANCH LENGTHS & COMPUTING H1 LIKELIHOOD")

edge_mlp_input = torch.cat([summary_token, e_dict[chosen_pair[0]], e_dict[chosen_pair[1]]], dim=0).unsqueeze(0)
edge_model = EdgeMLP()

with torch.no_grad():
    edge_logits = edge_model(edge_mlp_input).squeeze() # Shape [2500 bins]
    edge_probs = F.softmax(edge_logits, dim=0)
    
sampled_bin_idx = torch.multinomial(edge_probs, 1).item()

bin_t1 = sampled_bin_idx // 50
bin_t2 = sampled_bin_idx % 50
t1 = (bin_t1 + 1) * 0.01  
t2 = (bin_t2 + 1) * 0.01  

print(f"\n[SIMULATED] Edge MLP Sampling Chain executed:")
print(f"  -> Decoded Branch Lengths: t1 = {t1:.3f}, t2 = {t2:.3f}")

Q = jc_Q()
P1 = expm(Q * t1)
P2 = expm(Q * t2)

F_left = encoded[chosen_pair[0]]
F_right = encoded[chosen_pair[1]]
F_H1 = felsenstein_merge_step(F_left, F_right, P1, P2)

explain_tensor(
    "F_H1",
    np.round(F_H1, 4),
    "New Felsenstein partial likelihood matrix for the internal ancestor node H1.",
    "Computed recursively via Equation 1. It acts as the feature representation for H1 in ST1."
)

# ==========================================================
# QUESTION 2B — NEXT TRANSITION (ST1 -> ST2)
# ==========================================================
section("QUESTION 2B — NEXT TRANSITION: MERGE(H1, S4)")

y_H1 = torch.tensor(F_H1.flatten(), dtype=torch.float32)
e_H1 = seq_emb(y_H1)
e_S4 = e_dict["S4"]

e_s_st1 = torch.randn(128) # [SIMULATED] Next step context 
edge_mlp_input_st1 = torch.cat([e_s_st1, e_H1, e_S4], dim=0).unsqueeze(0)

with torch.no_grad():
    edge_logits_st1 = edge_model(edge_mlp_input_st1).squeeze()
    edge_probs_st1 = F.softmax(edge_logits_st1, dim=0)
    
sampled_bin_st1 = torch.multinomial(edge_probs_st1, 1).item()
t_H1_to_U = ((sampled_bin_st1 // 50) + 1) * 0.01  
t_S4_to_U = ((sampled_bin_st1 % 50) + 1) * 0.01  

P_H1 = expm(Q * t_H1_to_U)
P_S4 = expm(Q * t_S4_to_U)
F_S4 = encoded["S4"] 
F_U = felsenstein_merge_step(F_H1, F_S4, P_H1, P_S4)

explain_tensor(
    "F_U (The next ancestor)",
    np.round(F_U, 4),
    "Likelihood matrix for parent U = merge(H1, S4).",
    "Continues the recursive likelihood propagation up to the root."
)

# ==========================================================
# FINAL CAPSTONE: THE CONCEPTUAL MAP
# ==========================================================
section("FINAL CONCEPTUAL MAP SUMMARY: FROM BIOLOGY TO FINAL TREE")
print("""
Here is the step-by-step pipeline connecting the biological input to the mathematical execution:

1. Biological Sequences → Sequence Encoding: The raw DNA sequences (Σ={A,C,G,T}) are 
   ingested and converted into numerical formats using One-Hot Encoding.
2. Sequence Encoding → Leaf Features: These one-hot arrays mathematically act as the initial 
   Felsenstein features f_u^i, representing the conditional probability of observing the leaf.
3. Leaf Features → Forward Policy (P_F): The features are projected through linear embeddings 
   and a Transformer Encoder. The Transformer produces a global state token (e_s) and 
   contextualized tree embeddings (e_i, e_j). These are concatenated into a pairwise 
   state-action feature G_ij = [e_s ; e_i + e_j].
4. Forward Policy → Action Selection: A Multi-Layer Perceptron (SAMlp) maps the G_ij features 
   to scalar logits (l_ij). A Softmax function marginalizes these into a probability distribution. 
   The model samples one structural tree_action from this distribution.
5. Subtree Merging → Branch Lengths: An Edge MLP evaluates the chosen subtrees and the global 
   state to sample the edge_action, yielding continuous or discrete evolutionary times t1, t2.
6. Branch Lengths → Transition Matrices: The sampled times are applied to the Jukes-Cantor 
   rate matrix Q to compute the Markov transition matrices: P(t) = e^(Qt).
7. Transition Matrices → Felsenstein Likelihood: The environment recursively executes Felsenstein’s 
   Algorithm. It multiplies the child features by the transition matrices to compute the 
   partial likelihood feature of the newly formed ancestral node. 
8. Felsenstein Likelihood → Reward: Once a single tree remains, the final root likelihood 
   is combined with an exponential prior on the branch lengths to compute the final reward 
   R(z,b) = P(Y|z,b)P(b). The Trajectory Balance (TB) loss uses this to update weights.
9. Reward → Final Phylogenetic Tree Generation: The terminal graph is unrooted, yielding 
   the final inferred evolutionary hypothesis representing the biological lineages.
""")
