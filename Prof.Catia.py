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

def felsenstein_merge_step(F_left, F_right, P_left, P_right, verbose=False,
                            left_name="left", right_name="right", parent_name="parent"):
    """
This function uses Equation 1
from the PhyloGFN paper.

It can also show the calculation
site by site.

F_left and F_right
are the likelihood matrices
of the two child nodes.

Their shape is:

(n_sites, 4)

P_left and P_right
are the Jukes-Cantor
transition matrices
for the two branches.

Their shape is:

(4, 4)

The function returns F_parent.

F_parent is the likelihood matrix
for the new ancestor node.

Its shape is:

(n_sites, 4)
"""

    states = ["A", "C", "G", "T"]
    n_sites = F_left.shape[0]
    F_parent = np.zeros((n_sites, 4))

    if verbose:
        print(f"\n{'─'*80}")
        print(f"SITE-BY-SITE FELSENSTEIN TRACE  (parent = {parent_name}, "
              f"left = {left_name}, right = {right_name})")
        print(f"{'─'*80}")
        print("Formula (Equation 1):")
        print(f"  F_{parent_name}[site, a] = "
              f"[Σ_b  P_left[a,b]  · F_{left_name}[site,b]]")
        print(f"                       × [Σ_c  P_right[a,c] · F_{right_name}[site,c]]")
        print()

    for site in range(n_sites):
        for ps_idx, parent_state in enumerate(range(4)):
            left_sum  = np.sum(P_left[parent_state,  :] * F_left[site,  :])
            right_sum = np.sum(P_right[parent_state, :] * F_right[site, :])
            F_parent[site, parent_state] = left_sum * right_sum

        if verbose:
            left_obs  = states[np.argmax(F_left[site])]  if F_left[site].max()  == 1 else "mixed"
            right_obs = states[np.argmax(F_right[site])] if F_right[site].max() == 1 else "mixed"
            print(f"  Site {site+1}:  {left_name}={left_obs}  {right_name}={right_obs}")

            ps = 0
            ps_label = "A"
            lp = P_left[ps,  :] * F_left[site,  :]
            rp = P_right[ps, :] * F_right[site, :]
            ls = lp.sum()
            rs = rp.sum()
            print(f"    Worked example — parent_state = {ps_label}:")
            print(f"      P_left[{ps_label},:]  · F_{left_name}[site,:]  "
                  f"= {np.round(P_left[ps,:],4)} · {F_left[site,:]}")
            print(f"        = {np.round(lp,4)}  →  left_sum  = {ls:.6f}")
            print(f"      P_right[{ps_label},:] · F_{right_name}[site,:] "
                  f"= {np.round(P_right[ps,:],4)} · {F_right[site,:]}")
            print(f"        = {np.round(rp,4)}  →  right_sum = {rs:.6f}")
            print(f"      F_{parent_name}[site={site+1}, {ps_label}] = {ls:.6f} × {rs:.6f} "
                  f"= {ls*rs:.6f}")
            print(f"    Full row F_{parent_name}[site={site+1}, :] = {np.round(F_parent[site],6)}")
            print(f"    Biological meaning: F_{parent_name}[site={site+1}, A] = {F_parent[site,0]:.4f} "
                  f"means there is a {F_parent[site,0]*100:.2f}% probability that "
                  f"site {site+1} has state A at ancestor {parent_name}, given the observed "
                  f"nucleotides below ({left_name}={left_obs}, {right_name}={right_obs}).")
            print()

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
# SIMULATED vs DETERMINISTIC — OUTPUT LEGEND
# ==========================================================
section("OUTPUT LEGEND: SIMULATED vs DETERMINISTIC VALUES")
print("""
This script shows
the PhyloGFN pipeline.

It uses random
untrained network weights.

There are two types
of printed values:

[SIMULATED]

This value comes from
a random untrained
neural network.

In a trained model,
this value would be meaningful.

Here, it is only
an example.


[DETERMINISTIC]

This value is computed
from fixed inputs.

For example:

one-hot encoding,
matrix exponential,
or Felsenstein recursion.

These values are exact
and reproducible.

Every printed number
has one of these labels.
""")

# ==========================================================
# IMPORTANT LIMITATION OF THIS DEMONSTRATION
# ==========================================================
section("IMPORTANT NOTE: NO TRAINED MODEL IS USED")

print("""
This script shows
the PhyloGFN pipeline.

No trained model
is used.

All neural network
outputs are

[SIMULATED]

These include:

• Transformer outputs
• SAMlp logits
• EdgeMLP logits
• sampled actions
• sampled branch lengths

They are made
with random weights.


The following parts
are

[DETERMINISTIC]

• Felsenstein pruning
• Jukes-Cantor matrices
• reward computation

These follow
the paper exactly.


Pipeline:

DNA

↓

Embedding

↓

Transformer

↓

G_ij Construction

↓

Forward Policy

↓

Edge Policy

↓

Felsenstein Likelihood

↓

Reward Computation
""")



# ==========================================================
# STEP 0 & 1 — INPUT SEQUENCES & ONE-HOT ENCODING
# ==========================================================
S = {
    "S1": "AATG",
    "S2": "AATT",
    "S3": "ATGG",
    "S4": "ACGC",
}

section("STEP 0 & 1 — INITIAL STATE (ST0) & ONE-HOT ENCODING")
print("""
ST0 is the first state.

It has four separate
leaf nodes:

S1, S2, S3, and S4.

Each leaf contains
one DNA sequence.

One-hot encoding
changes each DNA letter

(A, C, G, T)

into a vector
with four values.

This gives

a (4 × 4) matrix

for each sequence.

This matrix is
the first Felsenstein
likelihood matrix.

The observed DNA state

has probability 1.

All other states

have probability 0.
""")

encoded = {name: seq2array(seq) for name, seq in S.items()}

explain_tensor(
    "F_S1 — [DETERMINISTIC] encoded leaf",
    encoded["S1"],
    "One-hot encoded matrix for sequence S1 ('AATG').",
    "Rows = sites (positions), Cols = DNA states {A,C,G,T}. "
    "Serves as the initial Felsenstein partial likelihood for leaf S1."
)

# ==========================================================
# STEP 2 — THE TOPOLOGICAL ACTION SPACE
# ==========================================================
section("STEP 2 — THE TOPOLOGICAL ACTION SPACE")

ST0 = list(S.keys())
actions = list(itertools.combinations(ST0, 2))

print(f"Number of trees in ST0 : {len(ST0)}")
print(f"C(4,2) = {len(actions)} candidate merge actions:\n")
for i, (a, b) in enumerate(actions, start=1):
    print(f"  a{i}: merge({a}, {b})")

# ==========================================================
# STEP 3 — TRANSFORMER EMBEDDINGS & CANONICAL G_ij VECTORS
# ==========================================================
section("STEP 3 — GENERATING CANONICAL G_ij VECTORS (Forward Policy Input)")

embedding_size = 128
input_size     = 16

seq_emb       = nn.Linear(input_size, embedding_size, bias=False)
encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=4, batch_first=True)
transformer   = nn.TransformerEncoder(encoder_layer, num_layers=1)

tree_embeddings_list = []
for name in ST0:
    y_i = torch.tensor(encoded[name].flatten(), dtype=torch.float32)
    tree_embeddings_list.append(seq_emb(y_i))

encoded_trees     = torch.stack(tree_embeddings_list).unsqueeze(0)   # (1, 4, 128)
h_s               = torch.randn(1, 1, embedding_size)
transformer_input = torch.cat([h_s, encoded_trees], dim=1)            # (1, 5, 128)

with torch.no_grad():
    transformer_output = transformer(transformer_input)

summary_token = transformer_output[0, 0, :]   # [SIMULATED]
e_dict = {
    "S1": transformer_output[0, 1, :],         # [SIMULATED]
    "S2": transformer_output[0, 2, :],         # [SIMULATED]
    "S3": transformer_output[0, 3, :],         # [SIMULATED]
    "S4": transformer_output[0, 4, :],         # [SIMULATED]
}

g_tensors = []
for (a, b) in actions:
    e_local = e_dict[a] + e_dict[b]
    g_ij    = torch.cat([summary_token, e_local], dim=0)
    g_tensors.append(g_ij)

g_batch = torch.stack(g_tensors)    # (6, 256)
G_S1S2  = g_batch[0]

print("""
[SIMULATED] Transformer embeddings are produced by a randomly initialised
Transformer encoder. In a trained model these would carry meaningful
phylogenetic context; here they are arbitrary 128-dimensional vectors.

Each G_ij feature vector is formed as:

  G_ij = [e_s ; e_i + e_j]   Shape: (256,) = 128 global + 128 local

  Indices   0–127  →  e_s          : global summary of the current MDP state
  Indices 128–255  →  e_i + e_j    : permutation-invariant encoding of the pair

The addition e_i + e_j is used instead of concatenation so that merge(S1,S2)
and merge(S2,S1) produce identical feature vectors, reflecting the fact that
merging is symmetric (the resulting tree topology is the same either way).
""")
print("  [SIMULATED] e_s  (global, indices 0–4):       ", G_S1S2[:5].detach().numpy())
print("  [SIMULATED] e_S1 + e_S2 (local, idx 128–132): ", G_S1S2[128:133].detach().numpy())
print(f"\n  Total dimension == 256?                    -> {len(G_S1S2) == 256}")
print(f"  Indices   0–127 match e_s exactly?         -> {torch.equal(G_S1S2[:128], summary_token)}")
print(f"  Indices 128–255 match e_S1 + e_S2 exactly? -> {torch.equal(G_S1S2[128:], e_dict['S1'] + e_dict['S2'])}")

# ==========================================================
# STEP 4 — LOGITS, SOFTMAX, GFLOWNET SAMPLING
# ==========================================================
section("STEP 4 — ACTION SCORING & GFLOWNET SAMPLING")

logits_head = SAMlp(in_features=256, hidden_features=256, out_features=1)

print("""
─────────────────────────────────────────────────────────────────────────────
SAMlp WHITE-BOX: full forward pass for G_S1S2 (action = merge(S1, S2))
─────────────────────────────────────────────────────────────────────────────

SAMlp is a two-layer MLP. The complete forward pass for a single pair (i,j) is:

    H    = GELU( G_ij @ W1.T + b1 )      hidden layer   (256-D)
    l_ij =       H    @ W2.T + b2        output scalar  (1-D)

Learnable parameters (fixed during the forward pass, updated by TB loss):
  W1  : shape (256, 256)   — maps 256-D input  → 256-D hidden
  b1  : shape (256,)       — bias of hidden layer
  W2  : shape (256, 1)     — maps 256-D hidden → 1 scalar
  b2  : shape (1,)         — bias of output layer

G_ij is the input that changes for every candidate pair; W1, b1, W2, b2 are
shared across all pairs within one forward pass and are updated across training
steps by minimising the Trajectory Balance (TB) loss.
""")

W1 = logits_head.fc1.weight   # (256, 256)
b1 = logits_head.fc1.bias     # (256,)
W2 = logits_head.fc2.weight   # (1, 256)
b2 = logits_head.fc2.bias     # (1,)

print(f"  W1 shape : {tuple(W1.shape)}")
print(f"  b1 shape : {tuple(b1.shape)}")
print(f"  W2 shape : {tuple(W2.shape)}")
print(f"  b2 shape : {tuple(b2.shape)}")

G_in = G_S1S2.detach()
with torch.no_grad():
    pre_act = G_in @ W1.T + b1          # (256,)
    H       = F.gelu(pre_act)           # (256,)
    l_S1S2  = (H @ W2.T + b2).item()   # scalar

print(f"""
[SIMULATED] Step-by-step computation for G_S1S2:

  1. pre_act = G_S1S2 @ W1.T + b1
       shape: ({G_in.shape[0]},) @ ({W1.T.shape[0]},{W1.T.shape[1]}) + ({b1.shape[0]},) → (256,)
       First 5 values of pre_act : {pre_act[:5].numpy().round(4)}

  2. H = GELU(pre_act)
       GELU(x) ≈ x·Φ(x)  (smooth approximation to ReLU)
       First 5 values of H       : {H[:5].numpy().round(4)}

  3. l_S1S2 = H @ W2.T + b2
       shape: (256,) @ (256,1) + (1,) → scalar
       l_S1S2 = {l_S1S2:.6f}

Range of l_ij: l_ij is an unbounded real number in (−∞, +∞). It is not a
probability and has no biological units. A large positive l_ij indicates that
SAMlp strongly favours this pair; a large negative value means it strongly
disfavours it. In this [SIMULATED] run l_S1S2 = {l_S1S2:.4f}.
""")

with torch.no_grad():
    logits        = logits_head(g_batch).squeeze(-1)   # (6,)
    probabilities = F.softmax(logits, dim=0)            # (6,)

print("""
─────────────────────────────────────────────────────────────────────────────
SOFTMAX FORMULA
─────────────────────────────────────────────────────────────────────────────

    P_F(a_k | ST0) = exp(l_k) / Σ_{j=1}^{6} exp(l_j)

  • l_k       = scalar logit from SAMlp for action k
  • exp(l_k)  = un-normalised weight (always positive)
  • Σ exp(l_j) = normalisation constant (sum over ALL 6 actions)

Softmax changes

the 6 logits

into 6 probabilities.

All probabilities

are greater than 0.

Their sum

is always 1.

The action

with the highest logit

gets the highest probability.

The other actions

still have

a chance

to be selected.

This lets

the model

explore

many different trees.

This is important

for GFlowNet.
─────────────────────────────────────────────────────────────────────────────
""")

print("[SIMULATED] Action → logit l_ij → Forward Probability P_F(a | ST0)")
for idx, (a, b) in enumerate(actions):
    print(f"  merge({a},{b})  logit = {logits[idx].item():+.4f}   "
          f"P_F = {probabilities[idx].item()*100:.2f}%")

prob_sum = probabilities.sum().item()
print(f"\n✓ VERIFICATION — sum of all 6 probabilities = {prob_sum:.6f}  (must equal 1.0000)")
assert abs(prob_sum - 1.0) < 1e-5, "Softmax probabilities do not sum to 1!"
print("  Assertion passed: softmax output is a valid probability distribution.\n")

argmax_idx    = torch.argmax(probabilities).item()
argmax_action = actions[argmax_idx]

sampled_idx   = torch.multinomial(probabilities, num_samples=1).item()
sampled_action = actions[sampled_idx]

print("""
─────────────────────────────────────────────────────────────────────────────
ARGMAX vs STOCHASTIC SAMPLING — concrete comparison
─────────────────────────────────────────────────────────────────────────────""")
print(f"  [SIMULATED] argmax would always choose : merge({argmax_action[0]}, {argmax_action[1]})"
      f"  (P = {probabilities[argmax_idx].item()*100:.2f}%)")
print(f"  [SIMULATED] sampling actually chose    : merge({sampled_action[0]}, {sampled_action[1]})"
      f"  (P = {probabilities[sampled_idx].item()*100:.2f}%)")
print(f"  Same choice this run?                  : {argmax_idx == sampled_idx}")

print("""
Why sampling

and not argmax?

GFlowNet

does not want

only one tree.

It wants

to learn

many possible trees.

The probability

of each tree

depends on

its reward

R(z,b).

If we use argmax,

the model

always chooses

the best score.

Then it may

find only one tree.

With sampling,

different trees

can be selected.

Trees with

higher reward

are selected

more often.

This helps

the model

explore

the full tree space.

During training,

the Trajectory Balance (TB)

objective learns

this behavior.
""")



chosen_left, chosen_right = "S1", "S2"
print(f"FIXED ACTION (professor's example): merge({chosen_left}, {chosen_right})")
print(f"  ST0 = {{ S1, S2, S3, S4 }}  →  ST1 = {{ H1, S3, S4 }}")

# ==========================================================
# STEP 5 — JUKES-CANTOR MODEL
# ==========================================================
section("STEP 5 — JUKES-CANTOR MODEL: Q MATRIX, P(t), AND BIOLOGICAL MEANING")

Q = jc_Q()

print("""
─────────────────────────────────────────────────────────────────────────────
THE JUKES-CANTOR RATE MATRIX Q   [DETERMINISTIC]
─────────────────────────────────────────────────────────────────────────────

Q is a 4×4 matrix governing instantaneous mutation rates between {A,C,G,T}.

    Q[a,b] = 1/3   for a ≠ b   (rate of mutating FROM state a TO state b)
    Q[a,a] = −1    (diagonal, ensures rows sum to zero)

The row-sum constraint is required for probability conservation: the master
equation dP/dt = Q·P must leave total probability equal to one. The diagonal
entry Q[A,A] = −1 equals minus the sum of all off-diagonal entries in that
row (1/3 + 1/3 + 1/3 = 1), meaning the total rate of leaving state A is 1.
""")
print("[DETERMINISTIC] Q matrix (rows = from-state, cols = to-state):")
print(Q)

row_sums = Q.sum(axis=1)
print(f"\n✓ VERIFICATION — row sums of Q: {np.round(row_sums, 10)}")
print(f"  All row sums ≈ 0?  -> {np.allclose(row_sums, 0.0)}")

print("""
─────────────────────────────────────────────────────────────────────────────
TRANSITION MATRIX P(t) = exp(Q·t)   [DETERMINISTIC]
─────────────────────────────────────────────────────────────────────────────

P(t)[a,b] is the probability that nucleotide b is observed after evolving for
branch length t, given the ancestor had nucleotide a. The Jukes-Cantor closed
form is:

    P(t)[same]      = 0.25 + 0.75 · exp(−4t/3)   (stay in same state)
    P(t)[different] = 0.25 − 0.25 · exp(−4t/3)   (mutate to any other state)
""")

# ==========================================================
# STEP 5b — EDGE MLP & BRANCH LENGTHS
# ==========================================================
section("STEP 5b — EDGE MLP: SAMPLING BRANCH LENGTHS")

edge_mlp_input = torch.cat([
    summary_token,
    e_dict[chosen_left],
    e_dict[chosen_right]
], dim=0).unsqueeze(0)

edge_model = EdgeMLP()

with torch.no_grad():
    edge_logits = edge_model(edge_mlp_input).squeeze()
    edge_probs  = F.softmax(edge_logits, dim=0)

sampled_bin = torch.multinomial(edge_probs, 1).item()
t1 = (sampled_bin // 50 + 1) * 0.01
t2 = (sampled_bin  % 50 + 1) * 0.01

print(f"\n[SIMULATED] EdgeMLP output → sampled bin index : {sampled_bin}")
print(f"[SIMULATED] Decoded branch lengths : t1 = {t1:.3f}  (branch to {chosen_left})")
print(f"[SIMULATED]                          t2 = {t2:.3f}  (branch to {chosen_right})")

P1 = expm(Q * t1)
P2 = expm(Q * t2)

print(f"""
─────────────────────────────────────────────────────────────────────────────
[DETERMINISTIC] BIOLOGICAL INTERPRETATION OF P(t) ENTRIES
─────────────────────────────────────────────────────────────────────────────

Transition matrix P(t1={t1:.3f}) for branch to {chosen_left}:
""")
print(np.round(P1, 4))

diag_val    = P1[0, 0]
offdiag_val = P1[0, 1]
print(f"""
  P(t1)[A,A] = {diag_val:.4f}
    → There is a {diag_val*100:.1f}% probability that nucleotide A remains A
      after evolving for branch length t1={t1:.3f}.

  P(t1)[A,C] = {offdiag_val:.4f}
    → There is a {offdiag_val*100:.1f}% probability that nucleotide A mutates to C
      (equivalently A→G or A→T — all off-diagonal entries are equal in the
      Jukes-Cantor model) after evolving for branch length t1={t1:.3f}.

  Row sum P(t1)[A,:] = {P1[0,:].sum():.6f}  (must equal 1.0 — verified below)
""")

print(f"✓ VERIFICATION — row sums of P(t1): {np.round(P1.sum(axis=1), 6)}")
print(f"  All row sums ≈ 1?  -> {np.allclose(P1.sum(axis=1), 1.0)}\n")

print(f"[DETERMINISTIC] Transition matrix P(t2={t2:.3f}) for branch to {chosen_right}:")
print(np.round(P2, 4))
print(f"\n  P(t2)[A,A] = {P2[0,0]:.4f}  → {P2[0,0]*100:.1f}% chance A stays A")
print(f"  P(t2)[A,C] = {P2[0,1]:.4f}  → {P2[0,1]*100:.1f}% chance A mutates to C")
print(f"\n✓ VERIFICATION — row sums of P(t2): {np.round(P2.sum(axis=1), 6)}")

lam = 10.0
print(f"""
─────────────────────────────────────────────────────────────────────────────
[DETERMINISTIC] EXPONENTIAL PRIOR ON BRANCH LENGTHS
─────────────────────────────────────────────────────────────────────────────

  P(t) = λ · exp(−λ · t)   with  λ = {lam:.0f}

  P(t1={t1:.3f}) = {lam:.0f} · exp(−{lam:.0f} × {t1:.3f}) = {lam * np.exp(-lam*t1):.6f}
  P(t2={t2:.3f}) = {lam:.0f} · exp(−{lam:.0f} × {t2:.3f}) = {lam * np.exp(-lam*t2):.6f}

  Biological meaning: long branches (large t) are exponentially penalised
  because they imply many mutations and risk of saturation (multiple hits at
  the same site obscure the true evolutionary distance). λ = {lam:.0f} means the
  prior expected branch length is 1/λ = {1/lam:.2f}.
─────────────────────────────────────────────────────────────────────────────
""")

# ==========================================================
# STEP 6 — FELSENSTEIN FOR H1
# ==========================================================
section("STEP 6 — FELSENSTEIN PRUNING: COMPUTING F_H1  (site-by-site trace)")

F_left  = encoded[chosen_left]
F_right = encoded[chosen_right]

print(f"[DETERMINISTIC] Left child  ({chosen_left}) one-hot matrix F_S1:\n{F_left}")
print(f"\n[DETERMINISTIC] Right child ({chosen_right}) one-hot matrix F_S2:\n{F_right}")

F_H1 = felsenstein_merge_step(
    F_left, F_right, P1, P2,
    verbose=True,
    left_name=chosen_left, right_name=chosen_right, parent_name="H1"
)

print("─" * 80)
print(f"\n[DETERMINISTIC] F_H1 — complete Felsenstein partial likelihood matrix for ancestor H1:")
print(f"  Shape: {F_H1.shape}  (rows=sites, cols=states {{A,C,G,T}})")
print(np.round(F_H1, 4))

print(f"""
Biological interpretation of F_H1: each entry F_H1[site, state] gives the
probability that ancestor H1 has nucleotide 'state' at 'site', given the
observed sequences of {chosen_left} and {chosen_right} below H1.

  Example: F_H1[0, A] = {F_H1[0,0]:.4f}
    → The probability that site 1 has state A at ancestor H1, given that
      {chosen_left} shows A and {chosen_right} shows A at site 1, integrated over
      all evolutionary paths along the two branches (t1={t1:.3f}, t2={t2:.3f}).

These are partial likelihoods (not marginals) because they will be multiplied
by further transition probabilities as we continue merging nodes toward the
root.
""")

# Collect branch lengths for prior computation
branch_lengths = [t1, t2]

# ==========================================================
# STEP 7 — SECOND MERGE: H1 + S4 → U
# ==========================================================
section("STEP 7 — SECOND MERGE: merge(H1, S4)  [ST1 → ST2]  (site-by-site trace)")

print("""
STATE TRANSITION:
  ST1 = { H1, S3, S4 }   (3 disjoint trees after first merge)
  Action: merge(H1, S4)
  ST2 = { U,  S3 }       (2 disjoint trees after second merge)

  H1 is not a DNA string. H1 is the (4-site × 4-state) Felsenstein likelihood
  matrix F_H1 computed in Step 6. It is flattened to a 16-dimensional vector,
  projected to 128 dimensions by seq_emb, and fed to the EdgeMLP exactly as
  a leaf sequence would be. This is how PhyloGFN handles internal nodes
  uniformly within the same architecture.
""")

# e_H1 is deterministic (computed from F_H1 via a fixed linear projection)
y_H1  = torch.tensor(F_H1.flatten(), dtype=torch.float32)
e_H1  = seq_emb(y_H1)          # [DETERMINISTIC] given fixed seq_emb weights
e_S4  = e_dict["S4"]           # [SIMULATED] from Transformer

# The summary token for ST1 would come from a fresh Transformer pass over ST1.
# We simulate it here because we have not built a full recurrent state encoder.
e_s_st1 = torch.randn(128)     # [SIMULATED] fresh summary for ST1

edge_mlp_input_st1 = torch.cat([e_s_st1, e_H1, e_S4], dim=0).unsqueeze(0)

with torch.no_grad():
    edge_logits_st1 = edge_model(edge_mlp_input_st1).squeeze()
    edge_probs_st1  = F.softmax(edge_logits_st1, dim=0)

sampled_bin_st1 = torch.multinomial(edge_probs_st1, 1).item()
t_H1 = (sampled_bin_st1 // 50 + 1) * 0.01
t_S4 = (sampled_bin_st1  % 50 + 1) * 0.01

print(f"[SIMULATED] Branch lengths: t_H1 = {t_H1:.3f} (H1→U)   t_S4 = {t_S4:.3f} (S4→U)")

P_H1 = expm(Q * t_H1)
P_S4 = expm(Q * t_S4)
F_S4 = encoded["S4"]

print(f"\n[DETERMINISTIC] P(t_H1={t_H1:.3f})[A,A] = {P_H1[0,0]:.4f}  "
      f"→ {P_H1[0,0]*100:.1f}% chance A stays A on H1→U branch")
print(f"[DETERMINISTIC] P(t_S4={t_S4:.3f})[A,A] = {P_S4[0,0]:.4f}  "
      f"→ {P_S4[0,0]*100:.1f}% chance A stays A on S4→U branch")

F_U = felsenstein_merge_step(
    F_H1, F_S4, P_H1, P_S4,
    verbose=True,
    left_name="H1", right_name="S4", parent_name="U"
)

branch_lengths += [t_H1, t_S4]

print("─" * 80)
print(f"\n[DETERMINISTIC] F_U — complete Felsenstein partial likelihood matrix for ancestor U:")
print(f"  Shape: {F_U.shape}  (rows=sites, cols=states {{A,C,G,T}})")
print(np.round(F_U, 4))

print(f"""
Biological interpretation of F_U: F_U[site, state] gives the probability
that ancestor U has nucleotide 'state' at 'site', given all observed sequences
below U (S1 and S2 via H1, and S4 directly).

  Example: F_U[0, A] = {F_U[0,0]:.4f}
    → The probability that site 1 has state A at ancestor U,
      given S1=A, S2=A, S4=A at site 1.

F_U will be used in the next merge step (U with S3 → final root), continuing
the bottom-up Felsenstein pruning toward the root.
""")

# ==========================================================
# STEP 8 — THIRD MERGE: U + S3 → ROOT  (final merge)
# ==========================================================
section("STEP 8 — THIRD MERGE: merge(U, S3) → ROOT  [ST2 → TERMINAL STATE]")

print("""
STATE TRANSITION:
  ST2 = { U, S3 }    (2 disjoint trees)
  Action: merge(U, S3)   — the only action available; no sampling needed.
  ST3 = { ROOT }     (terminal state — a single complete tree)

This is the final Felsenstein step. After it we have the root partial
likelihood matrix F_ROOT, which is used directly to compute the tree
likelihood P(Y | z, b) and the reward R(z, b).
""")

# Embed U similarly to H1
y_U  = torch.tensor(F_U.flatten(), dtype=torch.float32)
e_U  = seq_emb(y_U)            # [DETERMINISTIC] given fixed seq_emb weights
e_S3 = e_dict["S3"]            # [SIMULATED]

e_s_st2 = torch.randn(128)     # [SIMULATED] summary for ST2
edge_mlp_input_st2 = torch.cat([e_s_st2, e_U, e_S3], dim=0).unsqueeze(0)

with torch.no_grad():
    edge_logits_st2 = edge_model(edge_mlp_input_st2).squeeze()
    edge_probs_st2  = F.softmax(edge_logits_st2, dim=0)

sampled_bin_st2 = torch.multinomial(edge_probs_st2, 1).item()
t_U  = (sampled_bin_st2 // 50 + 1) * 0.01
t_S3 = (sampled_bin_st2  % 50 + 1) * 0.01

print(f"[SIMULATED] Branch lengths: t_U = {t_U:.3f} (U→ROOT)   t_S3 = {t_S3:.3f} (S3→ROOT)")

P_U  = expm(Q * t_U)
P_S3 = expm(Q * t_S3)
F_S3 = encoded["S3"]

F_ROOT = felsenstein_merge_step(
    F_U, F_S3, P_U, P_S3,
    verbose=True,
    left_name="U", right_name="S3", parent_name="ROOT"
)

branch_lengths += [t_U, t_S3]

print("─" * 80)
print(f"\n[DETERMINISTIC] F_ROOT — Felsenstein partial likelihood at the root:")
print(f"  Shape: {F_ROOT.shape}  (rows=sites, cols=states {{A,C,G,T}})")
print(np.round(F_ROOT, 4))

# ==========================================================
# STEP 9 — REWARD COMPUTATION  R(z, b) = P(Y|z,b) · P(b)
# ==========================================================
section("STEP 9 — REWARD COMPUTATION: R(z,b) = P(Y|z,b) · P(b)")

print("""
─────────────────────────────────────────────────────────────────────────────
[DETERMINISTIC] TREE LIKELIHOOD: P(Y | z, b)
─────────────────────────────────────────────────────────────────────────────

Under the Jukes-Cantor model, the stationary distribution is uniform:
  π(A) = π(C) = π(G) = π(T) = 0.25

The site likelihood for site k is:

  L_k = Σ_{c ∈ {A,C,G,T}}  π(c) · F_ROOT[k, c]

This marginalises over the unknown root state by weighting each state by its
stationary probability. The overall tree likelihood is the product across all
m sites (assuming independence):

  P(Y | z, b) = ∏_{k=1}^{m}  L_k
""")

pi = np.array([0.25, 0.25, 0.25, 0.25])   # uniform stationary distribution (JC)
n_sites = F_ROOT.shape[0]

site_likelihoods = np.zeros(n_sites)
for k in range(n_sites):
    site_likelihoods[k] = np.sum(pi * F_ROOT[k, :])
    print(f"  [DETERMINISTIC] L_{k+1} = Σ_c π(c)·F_ROOT[{k+1},c] "
          f"= 0.25 × {np.round(F_ROOT[k,:],6)} "
          f"= {site_likelihoods[k]:.8f}")

tree_likelihood = np.prod(site_likelihoods)
log_tree_likelihood = np.sum(np.log(site_likelihoods))

print(f"""
[DETERMINISTIC] P(Y | z, b) = L_1 × L_2 × L_3 × L_4
              = {' × '.join(f'{v:.6f}' for v in site_likelihoods)}
              = {tree_likelihood:.10e}

[DETERMINISTIC] log P(Y | z, b) = {log_tree_likelihood:.6f}
  (Log-space is preferred numerically to avoid floating-point underflow.)
""")

print("""
─────────────────────────────────────────────────────────────────────────────
[DETERMINISTIC] BRANCH LENGTH PRIOR: P(b)
─────────────────────────────────────────────────────────────────────────────

Each branch length t_k is assigned an independent Exponential(λ=10) prior:

  P(b) = ∏_k  λ · exp(−λ · t_k)
""")

lam = 10.0
log_prior = 0.0
for idx, t_k in enumerate(branch_lengths):
    lp_k = np.log(lam) - lam * t_k
    log_prior += lp_k
    print(f"  [DETERMINISTIC] Branch {idx+1}: t={t_k:.3f}  "
          f"log[λ·exp(−λt)] = log({lam:.0f}) − {lam:.0f}×{t_k:.3f} = {lp_k:.6f}")

prior = np.exp(log_prior)
print(f"""
[DETERMINISTIC] log P(b) = {log_prior:.6f}
[DETERMINISTIC] P(b)     = {prior:.10e}
""")

print("""
─────────────────────────────────────────────────────────────────────────────
[DETERMINISTIC] FINAL REWARD: R(z, b) = P(Y | z, b) · P(b)
─────────────────────────────────────────────────────────────────────────────

The GFlowNet reward is defined as:

  R(z, b) = P(Y | z, b) · P(b)

In log-space (numerically stable):

  log R(z, b) = log P(Y | z, b) + log P(b)
""")

log_reward = log_tree_likelihood + log_prior
reward     = np.exp(log_reward)

print(f"  [DETERMINISTIC] log P(Y|z,b)  = {log_tree_likelihood:.6f}")
print(f"  [DETERMINISTIC] log P(b)      = {log_prior:.6f}")
print(f"  [DETERMINISTIC] log R(z,b)    = {log_reward:.6f}")
print(f"  [DETERMINISTIC] R(z,b)        = {reward:.6e}")
print(f"""
Interpretation: R(z,b) = {reward:.4e} is the unnormalised posterior weight
assigned to this particular tree topology z and set of branch lengths b. A
higher R means this tree better explains the observed sequences (high
likelihood) while also satisfying the prior on branch lengths (not too long).

During GFlowNet training, the Trajectory Balance (TB) loss drives the forward
policy P_F to generate trees with frequency proportional to R(z,b), so that
after training the model approximates the Bayesian posterior P(z,b|Y) ∝ R(z,b).
""")

print("✓ VERIFICATION — all reward components are positive:")
print(f"  P(Y|z,b) > 0 : {tree_likelihood > 0}")
print(f"  P(b)     > 0 : {prior > 0}")
print(f"  R(z,b)   > 0 : {reward > 0}")

# ==========================================================
# FINAL CONCEPTUAL MAP SUMMARY
# ==========================================================
section("FINAL CONCEPTUAL MAP SUMMARY: FROM BIOLOGY TO FINAL TREE")

print("""
Step 1

DNA

↓

One-Hot Encoding

Each DNA sequence

is changed

into a matrix.

This matrix

is the first

Felsenstein likelihood.


Step 2

One-Hot

↓

Transformer Embeddings

Each matrix

is changed

into a

16-dimensional vector.

seq_emb changes it

into a

128-dimensional vector.

The Transformer

creates:

• e_s

  a global vector

  for the whole state.

• e_S1, e_S2,

  e_S3, e_S4

  one vector

  for each tree.


Step 3

Embeddings

↓

G_ij

G_ij = [e_s ; e_i + e_j]

The first part

contains

global information.

The second part

contains

the two candidate trees.


Step 4

G_ij

↓

Forward Policy

SAMlp gives

one logit

for each G_ij.

Softmax changes

the logits

into probabilities.

One action

is sampled.


Step 5

Chosen Action

↓

EdgeMLP

EdgeMLP uses

e_s,

e_i,

and e_j.

It samples

two branch lengths:

t1 and t2.


Step 6

Branch Lengths

↓

Transition Matrices

P(t) = exp(Q·t)

using

the Jukes-Cantor model.


Step 7

Transition Matrices

↓

Felsenstein Likelihood

Equation 1

computes

the new parent

likelihood.

The new parent

replaces

its two children.


Step 8

Terminal State

↓

Root Likelihood

After n−1 merges,

only one root

remains.

Its likelihood

is

P(Y | z,b).


Step 9

Reward

Reward

is

R(z,b)

=

P(Y | z,b)

×

P(b)

The reward

is used

to update

the model.


Step 10

Trained GFlowNet

↓

Final Tree

After training,

the model

generates

many trees.

Trees with

higher reward

are generated

more often.

The final result

is an

unrooted

bifurcating

phylogenetic tree.
""")
# ==========================================================
# MAIN IDEA OF THIS CODE
# ==========================================================
section("MAIN IDEA OF THIS CODE")

print("""
The main idea is simple.

First,

DNA is changed

into vectors.

Then,

GFlowNet chooses

which two trees

to merge.

After that,

EdgeMLP gives

the branch lengths.

Then,

Felsenstein calculates

the tree likelihood.

Finally,

the reward is computed.

The model learns

to generate trees

with higher probability

more often.
""")
