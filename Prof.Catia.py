import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4, sci_mode=False, edgeitems=4)
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
# HELPER FUNCTIONS & GLOBAL CLASSES  (defined ONCE)
# ==========================================================
def seq2array(seq):
    return np.array([MAP[x] for x in seq], dtype=np.float32)

def jc_Q():
    Q = np.ones((4, 4)) / 3.0
    np.fill_diagonal(Q, -1.0)
    return Q

def felsenstein_merge_step(F_left, F_right, P_left, P_right):
    """
    Implements Equation 1 from the PhyloGFN paper.
    F_left, F_right : (n_sites, 4) partial likelihood matrices of child nodes.
    P_left, P_right : (4, 4) Jukes-Cantor transition matrices for each branch.
    Returns F_parent : (n_sites, 4) partial likelihood for the new ancestor.
    """
    n_sites = F_left.shape[0]          # F_left.shape is (n_sites, 4); shape[0] = n_sites
    F_parent = np.zeros((n_sites, 4))
    for site in range(n_sites):
        for parent_state in range(4):
            left_prob  = np.sum(P_left[parent_state,  :] * F_left[site,  :])
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
    print("Shape:", tensor.shape if hasattr(tensor, "shape") else type(tensor))
    print("Role:", role)
    if isinstance(tensor, torch.Tensor):
        print(tensor.detach().numpy())
    else:
        print(tensor)

# SAMlp and EdgeMLP are each defined ONCE here.
class SAMlp(nn.Module):
    """Maps a G_ij feature vector (256-D) to a scalar logit l_ij."""
    def __init__(self, in_features=256, hidden_features=256, out_features=1, with_bias=True):
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
    """
    Takes a 384-D input (e_s || e_i || e_j, each 128-D) and outputs 2500 logits
    over a 50×50 joint bin grid for branch lengths (t1, t2).
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 2500)
        )

    def forward(self, x):
        return self.fc(x)

# ==========================================================
# IMPORTANT LIMITATION OF THIS DEMONSTRATION
# ==========================================================
section("IMPORTANT LIMITATION OF THIS DEMONSTRATION")
print("""
The original trained PhyloGFN checkpoint is not publicly available and is therefore
not loaded in this script.

Consequently, all Transformer outputs, SAMlp logits, EdgeMLP outputs, sampled
actions, and sampled branch lengths are generated using randomly initialised
(untrained) network parameters.

The purpose of this script is NOT to reproduce the numerical predictions of the
published model.

Its purpose is to demonstrate the computational pipeline and mathematical
operations described in the PhyloGFN paper:

DNA
→ Embedding
→ Transformer
→ G_ij Construction
→ Forward Policy
→ Edge Policy
→ Felsenstein Likelihood
→ Reward Computation

Therefore, the model structure, tensor sizes, mathematical equations, and data flow follow the paper, while the neural network values are only [SIMULATED] examples.
""")

# ==========================================================
# STEP 0 & 1 — INPUT SEQUENCES & ONE-HOT ENCODING
# ==========================================================
# Naming convention (strictly observed throughout):
#   S1, S2, S3, S4  →  biological leaf sequences (fixed labels).
#   ST0, ST1, ST2   →  MDP states (number of disjoint trees shrinks at each step).
S = {
    "S1": "AATG",
    "S2": "AATT",
    "S3": "ATGG",
    "S4": "ACGC",
}

section("STEP 0 & 1 — INITIAL STATE (ST0) & ONE-HOT ENCODING")
print("ST0 = Initial State. ST0 contains four disjoint leaf nodes: S1, S2, S3, S4.")
print("S1–S4 are the biological sequences. ST0 is the MDP state (not a sequence label).\n")

encoded = {name: seq2array(seq) for name, seq in S.items()}

explain_tensor(
    "F_S1 (example encoded leaf)",
    encoded["S1"],
    "One-hot encoded matrix for sequence S1 ('AATG').",
    "Rows = sites (positions), Cols = DNA states {A,C,G,T}. "
    "Serves as the initial Felsenstein partial likelihood for leaf S1."
)

print("\n" + "-" * 80)
print("THEORETICAL CONTEXT: DNA vs MLST")
print("-" * 80)
print("""In standard PhyloGFN, the initial feature for leaf u at site i is f_u^i ∈ [0,1]^4.
For MLST (Multi-Locus Sequence Typing), the alphabet shifts to allele IDs (1..K).
The vocabulary size expands from 4 to K, changing the state space from m×4 to m×K.
For cgMLST (Core Genome), each locus needs its own rate matrix Q_i and scaler μ_i:
   P_i(t) = exp(Q_i * μ_i * t).""")

# ==========================================================
# EXPLANATION — ACTION SPACE & TREE CONSTRUCTION
# ==========================================================
section("EXPLANATION — ACTION SPACE & TREE CONSTRUCTION")
print("""1. Action Space Definition
In PhyloGFN the state is a set of disjoint rooted trees. At any non-terminal state with
l > 1 trees, one transition = two parts:
  • Tree Action (topological merge): choose which two subtrees to join.
    There are C(l,2) choices; l shrinks by 1 each step.
  • Edge Action (branch lengths): the Edge MLP jointly samples branch lengths (t1, t2)
    from a discrete 50×50 bin grid (multinomial) or a Gaussian mixture (continuous).

2. Tree Construction Process (bottom-up MDP)
  ST0 : n disconnected leaf nodes (one per sequence).
  Each step: merge two roots → new internal node reduces tree count by 1.
  After n−1 steps: one rooted tree remains.
  Because Jukes-Cantor is time-reversible the root is dropped, yielding an
  unrooted bifurcating tree.""")

# ==========================================================
# STEP 2 — THE TOPOLOGICAL ACTION SPACE
# ==========================================================
section("STEP 2 — THE TOPOLOGICAL ACTION SPACE")

ST0 = list(S.keys())           # ['S1', 'S2', 'S3', 'S4']
actions = list(itertools.combinations(ST0, 2))   # C(4,2) = 6 pairs

print(f"Number of trees in ST0 : {len(ST0)}")
print(f"C(4,2) = {len(actions)} candidate merge actions:\n")
for i, (a, b) in enumerate(actions, start=1):
    print(f"  a{i}: merge({a}, {b})")
section("MATHEMATICAL DERIVATION OF G_ij")
print("""
The Forward Policy does not operate directly on raw DNA sequences.

For each leaf sequence S_k:

DNA sequence
→ One-Hot Encoding
→ Flattening (4×4 → 16-D)
→ Linear Projection (seq_emb : R^16 → R^128)
→ Initial Tree Embedding

The embeddings of all trees in the current state ST0 are then passed through a
Transformer Encoder.

Transformer Input:
[h_s ; e_S1 ; e_S2 ; e_S3 ; e_S4]

where:
h_s  = learnable summary query token
e_Sk = embedding of tree S_k

The Transformer outputs:
[e_s ; e'_S1 ; e'_S2 ; e'_S3 ; e'_S4]

where:
e_s   = global summary token encoding the entire state ST0
e'_Si = contextualized embedding of tree i

For a candidate merge action (i,j), the canonical state-action representation is:

G_ij = [e_s ; e'_i + e'_j]

The first 128 dimensions encode global context.
The last 128 dimensions encode the candidate pair.

Thus:

DNA
→ One-Hot
→ seq_emb
→ Transformer
→ e_s, e_i, e_j
→ G_ij
→ SAMlp
→ logits
→ softmax
→ P_F(a | ST0)
""")
# ==========================================================
# STEP 3 — TRANSFORMER EMBEDDINGS & CANONICAL G_ij VECTORS
# ==========================================================
section("STEP 3 — GENERATING CANONICAL G_ij VECTORS (Forward Policy Input)")

print("""
[WHAT IS G_ij?]
G_ij is the feature vector the Forward Policy uses to score the action "merge tree i with
tree j". It is constructed by concatenating two parts:

  G_ij  =  [ e_s  ;  e_i + e_j ]        (Appendix D, PhyloGFN paper)

  • e_s      (indices   0–127, 128 features): Global summary token produced by the
              Transformer. It encodes the *entire current state* ST0 in an
              order-equivariant way (the summary token attends to all trees at once).

  • e_i+e_j  (indices 128–255, 128 features): Elementwise sum of the two candidate
              tree embeddings. Summing (instead of concatenating) keeps G_ij
              permutation-invariant: merge(S1,S2) = merge(S2,S1).

  Total: 128 + 128 = 256 features per candidate action.

All six G_ij vectors are stacked into a (6, 256) batch and fed to SAMlp, which maps
each to a scalar logit. Softmax over the 6 logits gives P_F(a | ST0).
""")

embedding_size = 128
input_size     = 16      # 4 sites × 4 DNA states (flattened one-hot)

seq_emb       = nn.Linear(input_size, embedding_size, bias=False)
encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=4, batch_first=True)
transformer   = nn.TransformerEncoder(encoder_layer, num_layers=1)

# Build transformer input: [summary_slot | e_S1 | e_S2 | e_S3 | e_S4]
# h_s is a learnable summary query; here it is randomly initialised ([SIMULATED]).
tree_embeddings_list = []
for name in ST0:
    y_i = torch.tensor(encoded[name].flatten(), dtype=torch.float32)
    tree_embeddings_list.append(seq_emb(y_i))

encoded_trees   = torch.stack(tree_embeddings_list).unsqueeze(0)   # (1, 4, 128)
h_s             = torch.randn(1, 1, embedding_size)                # [SIMULATED] summary query
transformer_input = torch.cat([h_s, encoded_trees], dim=1)         # (1, 5, 128)

with torch.no_grad():
    transformer_output = transformer(transformer_input)            # (1, 5, 128)

summary_token = transformer_output[0, 0, :]       # e_s : (128,)
e_dict = {
    "S1": transformer_output[0, 1, :],
    "S2": transformer_output[0, 2, :],
    "S3": transformer_output[0, 3, :],
    "S4": transformer_output[0, 4, :],
}

# Build the (6, 256) G_ij batch — ALWAYS global first, local second (paper convention)
g_tensors = []
for (a, b) in actions:
    e_local = e_dict[a] + e_dict[b]                          # permutation-invariant sum
    g_ij    = torch.cat([summary_token, e_local], dim=0)     # [global(128) ; local(128)]
    g_tensors.append(g_ij)

g_batch = torch.stack(g_tensors)    # (6, 256)
print("""
IMPORTANT NOTE ABOUT THE NUMBERS BELOW

The following tensor contains 256 learned latent features.

Indices 0–127:
Global state representation e_s.

Indices 128–255:
Local candidate-pair representation e_S1 + e_S2.

The individual numerical values do not have a direct biological interpretation.
Only the complete vector is used by the neural network.

SAMlp processes this 256-dimensional representation and learns how combinations
of these features correlate with high-reward phylogenetic constructions.

The tensor is printed only to verify the exact structure of G_ij and to
demonstrate that the implementation matches the mathematical definition:

G_ij = [e_s ; e_i + e_j]
""")
# Show ONE canonical example with full decomposition proof
G_S1S2 = g_batch[0]
explain_tensor(
    "G_S1S2 — canonical feature vector for action merge(S1, S2)",
    G_S1S2,
    "256-D vector: indices 0–127 = e_s (global state), indices 128–255 = e_S1 + e_S2 (local pair).",
    "Formula: G_ij = [e_s ; e_i + e_j]. One such vector exists for each of the 6 candidate actions."
)

print("\n[VERIFYING THE FEATURE DECOMPOSITION FOR G_S1S2]")
print(f"  Total dimension == 256?                   -> {len(G_S1S2) == 256}")
print(f"  Indices   0–127 match e_s exactly?        -> {torch.equal(G_S1S2[:128], summary_token)}")
print(f"  Indices 128–255 match e_S1 + e_S2 exactly?-> {torch.equal(G_S1S2[128:], e_dict['S1'] + e_dict['S2'])}")

# ==========================================================
# EXPLANATION — FORWARD POLICY & THE "WHY"
# ==========================================================
section("EXPLANATION — FORWARD POLICY & THE 'WHY'")
print("""
3. Forward Policy vs. Random Actions
  • Learned P_F: SAMlp maps each G_ij to a scalar logit. Softmax converts the 6 logits into
    a probability distribution. The generator *samples* from this distribution rather than
    always taking the greedy maximum, which is essential for exploration.
  • ε-greedy exploration: during training PhyloGFN occasionally samples a uniformly random
    action with probability ε to prevent mode collapse.

4. Why GFlowNet?
  The space of unrooted bifurcating trees is super-exponential — (2n−5)!! for n leaves —
  so exhaustive search is impossible. The GFlowNet objective trains P_F to generate trees
  proportionally to their posterior reward R(z,b) = P(Y|z,b)·P(b). This amortised
  posterior sampler efficiently discovers diverse high-reward trees, which is biologically
  essential for computing branch support and confidence intervals.""")

# ==========================================================
# STEP 4 — LOGITS, SOFTMAX, AND GFLOWNET SAMPLING
# ==========================================================
section("STEP 4 — ACTION SCORING & GFLOWNET SAMPLING")

logits_head = SAMlp(in_features=256, hidden_features=256, out_features=1)

with torch.no_grad():
    logits        = logits_head(g_batch).squeeze(-1)   # (6,)
    probabilities = F.softmax(logits, dim=0)           # (6,)  sums to 1

print("\n[Action → Forward Probability P_F(a | ST0)]")
for idx, (a, b) in enumerate(actions):
    print(f"  P_F(merge({a}, {b})) = {probabilities[idx].item()*100:.2f}%"
          f"  (logit: {logits[idx].item():.4f})")

# Stochastic sampling proportional to P_F  ([SIMULATED] because weights are untrained)
sampled_idx  = torch.multinomial(probabilities, num_samples=1).item()
chosen_left, chosen_right = actions[sampled_idx]

print(f"\n{'='*80}")
print(f"[SIMULATED] SAMPLED ACTION: merge({chosen_left}, {chosen_right})")
print(f"{'='*80}")
print(f"State transition: ST0 → ST1")
print(f"  The environment merges sequence {chosen_left} and sequence {chosen_right}.")
print(f"  A new internal ancestral node H1 replaces them; ST1 now has 3 disjoint trees.")

# ==========================================================
# EXPLANATION — EVOLUTIONARY MODELS & FELSENSTEIN PRUNING
# ==========================================================
section("EXPLANATION — EVOLUTIONARY MODELS & FELSENSTEIN PRUNING")
print("""
1. Jukes-Cantor (JC) Model
   Instantaneous rate matrix Q (4×4): off-diagonal = 1/3, diagonal = −1.
   Governs: dP(t)/dt = Q · P(t).

2. Transition Matrices
   Branch length t = evolutionary distance. P(t) = exp(Q·t) via matrix exponentiation.
   In efficient code this uses spectral decomposition: exp(Qt) = U · exp(Dt) · U⁻¹.

3. Exponential Prior on Branch Lengths
   P(t) = λ · exp(−λ·t) with λ=10. Biologically, this penalises unrealistically long
   branches (saturation) and acts as a regulariser during training.

4. Felsenstein Pruning Algorithm (Equation 1)
   For internal node u with children v, w:
     F_u[i, a_u] = [Σ_{a_v} P(a_v|a_u, t_v) · F_v[i, a_v]]
                 × [Σ_{a_w} P(a_w|a_u, t_w) · F_w[i, a_w]]
   'Partial' because it conditions on a specific state a_u at node u.""")

# ==========================================================
# QUESTION 2A — EDGE MLP SAMPLING & H1 FELSENSTEIN MATRIX
# ==========================================================
section("QUESTION 2A — SAMPLING BRANCH LENGTHS & COMPUTING H1 LIKELIHOOD")

print(f"\nChosen merge: ({chosen_left}, {chosen_right})  →  new node H1")
print("""
Branch lengths are NOT hardcoded. Here is the full EdgeMLP sampling chain:

  Step A: Concatenate [e_s (128-D) || e_left (128-D) || e_right (128-D)] → 384-D input.
  Step B: EdgeMLP maps 384-D → 2500 logits (a 50×50 joint bin grid for t1 and t2).
  Step C: Softmax → probability distribution over 2500 bins.
  Step D: torch.multinomial samples one bin index  [SIMULATED].
  Step E: Decode:  t1 = (bin_index // 50 + 1) × 0.01
                   t2 = (bin_index  % 50 + 1) × 0.01
          This gives branch lengths in the range [0.01, 0.50].
""")

# Build EdgeMLP input for ST0 → ST1 transition
edge_mlp_input = torch.cat([
    summary_token,
    e_dict[chosen_left],
    e_dict[chosen_right]
], dim=0).unsqueeze(0)            # (1, 384)

print(f"EdgeMLP input shape: {edge_mlp_input.shape}  "
      f"= [e_s(128) || e_{chosen_left}(128) || e_{chosen_right}(128)]")

edge_model = EdgeMLP()

with torch.no_grad():
    edge_logits = edge_model(edge_mlp_input).squeeze()   # (2500,)  [SIMULATED]
    edge_probs  = F.softmax(edge_logits, dim=0)

sampled_bin = torch.multinomial(edge_probs, 1).item()    # [SIMULATED]
t1 = (sampled_bin // 50 + 1) * 0.01
t2 = (sampled_bin  % 50 + 1) * 0.01

print(f"\n[SIMULATED] EdgeMLP output → sampled bin index : {sampled_bin}")
print(f"[SIMULATED] Decoded branch lengths : t1 = {t1:.3f}  (branch to {chosen_left})")
print(f"[SIMULATED]                          t2 = {t2:.3f}  (branch to {chosen_right})")

# Jukes-Cantor transition matrices
Q  = jc_Q()
P1 = expm(Q * t1)
P2 = expm(Q * t2)

print(f"\nJukes-Cantor rate matrix Q (4×4):\n{np.round(Q, 4)}")
print(f"\nTransition matrix P(t1={t1:.3f}) = exp(Q·t1):\n{np.round(P1, 4)}")
print(f"\nTransition matrix P(t2={t2:.3f}) = exp(Q·t2):\n{np.round(P2, 4)}")

# Felsenstein recursion
F_left  = encoded[chosen_left]
F_right = encoded[chosen_right]
F_H1    = felsenstein_merge_step(F_left, F_right, P1, P2)

explain_tensor(
    "F_H1 — Felsenstein partial likelihood for ancestral node H1",
    np.round(F_H1, 4),
    f"H1 is the new ancestor of {chosen_left} and {chosen_right}.",
    "Each entry F_H1[site, state] = probability that site has 'state' at H1 given the "
    "observed leaves below. Computed via Equation 1 (Felsenstein recursion)."
)

# ==========================================================
# QUESTION 2B — NEXT TRANSITION: merge(H1, S4)  →  ST1 → ST2
# ==========================================================
section("QUESTION 2B — NEXT TRANSITION: MERGE(H1, S4)")

print("""
State ST1 contains three disjoint trees: H1, S2/S3 (whichever was not chosen), and the
remaining leaf. For the professor's concrete example we now perform merge(H1, S4).

The procedure is identical to Question 2A but the left child is H1 (an internal node
whose Felsenstein likelihood F_H1 was just computed) instead of a raw leaf.

H1's embedding e_H1 is obtained by projecting F_H1.flatten() through seq_emb
(the same linear layer used for leaves). This is how the Transformer reincorporates
newly formed ancestors back into the state representation.

In a real trained model, the Transformer would be re-run on ST1 to obtain a fresh
summary token. Here we use torch.randn to simulate that token  [SIMULATED].
""")

# Embed H1 via the same seq_emb projection (F_H1 is (4,4), flatten to 16-D)
y_H1  = torch.tensor(F_H1.flatten(), dtype=torch.float32)   # (16,)
e_H1  = seq_emb(y_H1)                                       # (128,) — deterministic given F_H1
e_S4  = e_dict["S4"]

# [SIMULATED] fresh summary token for ST1 (would require re-running Transformer)
e_s_st1 = torch.randn(128)    # [SIMULATED]

edge_mlp_input_st1 = torch.cat([e_s_st1, e_H1, e_S4], dim=0).unsqueeze(0)  # (1, 384)
print(f"EdgeMLP input shape for ST1 → ST2 transition: {edge_mlp_input_st1.shape}")

with torch.no_grad():
    edge_logits_st1 = edge_model(edge_mlp_input_st1).squeeze()   # [SIMULATED]
    edge_probs_st1  = F.softmax(edge_logits_st1, dim=0)

sampled_bin_st1 = torch.multinomial(edge_probs_st1, 1).item()    # [SIMULATED]
t_H1 = (sampled_bin_st1 // 50 + 1) * 0.01
t_S4 = (sampled_bin_st1  % 50 + 1) * 0.01

print(f"\n[SIMULATED] Sampled bin index : {sampled_bin_st1}")
print(f"[SIMULATED] Branch lengths    : t_H1 = {t_H1:.3f}  (branch H1 → U)")
print(f"[SIMULATED]                     t_S4 = {t_S4:.3f}  (branch S4 → U)")

P_H1 = expm(Q * t_H1)
P_S4 = expm(Q * t_S4)
F_S4 = encoded["S4"]
F_U  = felsenstein_merge_step(F_H1, F_S4, P_H1, P_S4)

explain_tensor(
    "F_U — Felsenstein partial likelihood for ancestor U = merge(H1, S4)",
    np.round(F_U, 4),
    "U is the new ancestor joining H1 and S4.",
    "Computed by applying Equation 1 to F_H1 and F_S4 with their respective "
    "transition matrices. This continues the recursive likelihood propagation toward the root."
)

# ==========================================================
# FINAL CONCEPTUAL MAP SUMMARY
# ==========================================================
section("FINAL CONCEPTUAL MAP SUMMARY: FROM BIOLOGY TO FINAL TREE")
print("""
Step 1  Biological Sequences → One-Hot Encoding
        Raw DNA (Σ={A,C,G,T}) is converted to binary indicator matrices F_leaf ∈ {0,1}^(m×4).

Step 2  One-Hot Leaves → Transformer Embeddings
        seq_emb (linear) projects each (16,)-flat leaf to a 128-D vector.
        A Transformer Encoder contextualises all leaf embeddings jointly, producing:
          • e_s        : 128-D global summary token (encodes full state ST0).
          • e_S1..e_S4 : 128-D individual tree embeddings.

Step 3  Embeddings → G_ij Feature Vectors
        For each candidate pair (i,j): G_ij = [e_s ; e_i + e_j]  ∈ R^256.
        The global part captures context; the local part is permutation-invariant.

Step 4  G_ij → Action Sampling via Forward Policy P_F
        SAMlp maps each G_ij → scalar logit l_ij.
        Softmax(logits) → P_F(a | ST). One action is sampled ∝ P_F.

Step 5  Chosen Action → Branch Lengths via EdgeMLP
        EdgeMLP(e_s || e_i || e_j) → 2500 bin logits → multinomial sample → (t1, t2).

Step 6  Branch Lengths → Transition Matrices
        P(t) = exp(Q·t)  under the Jukes-Cantor model.

Step 7  Transition Matrices → Felsenstein Likelihood (Equation 1)
        F_parent[site, a_u] = [Σ P(a_v|a_u,t1)·F_left[site,a_v]]
                             × [Σ P(a_w|a_u,t2)·F_right[site,a_w]]
        The new ancestor's likelihood replaces its children in the next state.

Step 8  Terminal State → Reward
        After n−1 merges, one tree remains. The reward is:
          R(z,b) = P(Y|z,b) · P(b)
        where P(Y|z,b) = product over sites of root likelihood summed over root states,
        and P(b) = ∏ λ·exp(−λ·t_k) is the exponential prior on all branch lengths.
        The Trajectory Balance (TB) loss uses R to update all network weights.

Step 9  Reward → Proportional Tree Generation
        After training, the GFlowNet generates each tree topology z with branch lengths b
        with probability exactly proportional to R(z,b), giving a correct posterior sample.
        The rooted tree is then unrooted (Jukes-Cantor reversibility) to yield the final
        unrooted bifurcating phylogenetic tree.
""")
