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
    "A": [1., 0., 0., 0.],
    "C": [0., 1., 0., 0.],
    "G": [0., 0., 1., 0.],
    "T": [0., 0., 0., 1.],
    "N": [1., 1., 1., 1.],
}


def seq2array(seq):
    return np.array([MAP[x] for x in seq], dtype=float)


def jc_Q():
    Q = np.ones((4, 4)) / 3.0
    np.fill_diagonal(Q, -1.0)
    return Q


def felsenstein_merge(F_left, F_right, P_left, P_right):
    n_sites = F_left.shape[0]
    F_parent = np.zeros((n_sites, 4))

    for site in range(n_sites):
        for parent_state in range(4):
            left = np.sum(P_left[parent_state, :] * F_left[site, :])
            right = np.sum(P_right[parent_state, :] * F_right[site, :])
            F_parent[site, parent_state] = left * right

    return F_parent


def show(name, x):
    print(f"\n{name}")
    print(f"shape = {x.shape}")
    print(x)


# ==========================================================
# STEP 0 — Input sequences
# ==========================================================
S = {
    "S1": "AATG",
    "S2": "AATT",
    "S3": "ATGG",
    "S4": "ACGC",
}

print("=" * 80)
print("STEP 0 — INPUT SEQUENCES")
print("=" * 80)

for name, seq in S.items():
    print(f"{name} = {seq}")


# ==========================================================
# STEP 1 — One-hot encoding
# ==========================================================
print("\n" + "=" * 80)
print("STEP 1 — ONE-HOT ENCODING")
print("=" * 80)

encoded = {}

for name, seq in S.items():
    encoded[name] = seq2array(seq)
    show(f"F_{name} / encoded {name}", encoded[name])

print("\nMeaning:")
print("Rows = sequence sites.")
print("Columns = DNA states [A, C, G, T].")
print("Each leaf sequence has shape (4,4).")


# ==========================================================
# QUESTION 1 — Select first merge action
# ==========================================================
print("\n" + "=" * 80)
print("QUESTION 1 — HOW SELECT FIRST MERGE ACTION?")
print("=" * 80)

state_0 = list(S.keys())
actions = list(itertools.combinations(state_0, 2))

print(f"\nInitial state S0 = {state_0}")
print(f"Number of possible actions = C(4,2) = {len(actions)}")

for i, pair in enumerate(actions):
    print(f"a{i+1} = merge{pair}")


# ==========================================================
# STEP 2 — Build G_ij demonstrative tensors
# ==========================================================
print("\n" + "=" * 80)
print("STEP 2 — BUILD G_ij FOR EACH POSSIBLE ACTION")
print("=" * 80)

for i, (a, b) in enumerate(actions):
    flat = np.concatenate([encoded[a].flatten(), encoded[b].flatten()])
    rows = np.concatenate([encoded[a], encoded[b]], axis=0)
    cols = np.concatenate([encoded[a], encoded[b]], axis=1)

    print(f"\nAction a{i+1} = merge({a},{b})")
    show(f"G_{a}{b}_flat = concat(flatten({a}), flatten({b}))", flat)
    show(f"G_{a}{b}_rows = concat rows", rows)
    show(f"G_{a}{b}_cols = concat columns", cols)

print("\nIMPORTANT:")
print("These G_ij formats are explanatory.")
print("The real PhyloGFN G_ij must be printed from the actual model code.")


# ==========================================================
# STEP 3 — Real neural-network-style MLP trace
# ==========================================================
print("\n" + "=" * 80)
print("STEP 3 — NEURAL NETWORK STYLE LOGITS")
print("=" * 80)

print("""
Important warning:
This section does NOT load the trained PhyloGFN checkpoint.
It demonstrates the same mathematical mechanism:
G_ij -> MLP -> logits -> softmax -> selected action.

The embeddings below are simulated Transformer outputs.
For a perfect professor answer, these must be replaced by real prints from Transformer.forward().
""")

embedding_size = 128
num_trees = 4

# Simulated Transformer outputs:
# In real PhyloGFN, these come from:
# x = self.seq_emb(batch_input)
# x = encoder(x, key_padding_mask)
encoded_trees = torch.randn(num_trees, embedding_size)
summary_token = torch.randn(embedding_size)

tree_indices = list(range(num_trees))
tree_pairs_idx = list(itertools.combinations(tree_indices, 2))

g_tensors = []

for i, j in tree_pairs_idx:
    combined_tree_rep = encoded_trees[i] + encoded_trees[j]
    g_ij = torch.cat([summary_token, combined_tree_rep], dim=0)
    g_tensors.append(g_ij)

g_batch = torch.stack(g_tensors)

print(f"Simulated Transformer tree embeddings shape: {encoded_trees.shape}")
print(f"Summary token shape: {summary_token.shape}")
print(f"G batch shape = {g_batch.shape}")
print("Meaning: 6 actions, each G_ij has 256 features = [summary_token ; e_i + e_j]")

print("\nExample G_12:")
print(f"G_12 shape = {g_batch[0].shape}")
print(g_batch[0])

part1_logits_head = nn.Sequential(
    nn.Linear(embedding_size * 2, 256),
    nn.GELU(),
    nn.Linear(256, 1),
)

with torch.no_grad():
    logits = part1_logits_head(g_batch).squeeze(-1)

print("\nCalculated logits from MLP:")
for idx, (i, j) in enumerate(tree_pairs_idx):
    print(f"L_{i+1}{j+1} = {logits[idx].item():.6f}")

probabilities = F.softmax(logits, dim=0)

print("\nSoftmax probabilities:")
for idx, (i, j) in enumerate(tree_pairs_idx):
    print(f"P(merge S{i+1},S{j+1} | S0) = {probabilities[idx].item():.6f}")

chosen_action_idx = torch.argmax(probabilities).item()
chosen_pair = tree_pairs_idx[chosen_action_idx]

print("\nSelected first merge by argmax:")
print(f"action index = a{chosen_action_idx + 1}")
print(f"selected pair = merge(S{chosen_pair[0] + 1}, S{chosen_pair[1] + 1})")

print("\nFor real PhyloGFN, print these inside the real code:")
print("print(g12.shape)")
print("print(g12)")
print("print(logits.shape)")
print("print(logits)")
print("print(tree_actions)")


# ==========================================================
# QUESTION 2 PART A — Calculate H1 = merge(S1,S2)
# ==========================================================
print("\n" + "=" * 80)
print("QUESTION 2 PART A — CALCULATE H1 = MERGE(S1,S2)")
print("=" * 80)

Q = jc_Q()
show("Jukes-Cantor Q matrix", Q)
print("Row sums:", Q.sum(axis=1))

t1 = 0.1
t2 = 0.1

P1 = expm(Q * t1)
P2 = expm(Q * t2)

show(f"P(t1={t1})", P1)
show(f"P(t2={t2})", P2)

F_S1 = encoded["S1"]
F_S2 = encoded["S2"]

F_H1 = felsenstein_merge(F_S1, F_S2, P1, P2)

show("F_H1 = likelihood matrix for internal node H1", F_H1)

print("\nInterpretation:")
print("Rows = sites 1..4")
print("Columns = possible parent states [A, C, G, T]")
print("H1 is NOT a DNA letter.")
print("H1 is an internal node represented by this likelihood matrix.")

for site in range(4):
    print(f"\nSite {site+1}: S1={S['S1'][site]}, S2={S['S2'][site]}")
    for j, state in enumerate(DNA):
        print(f"  F_H1[site {site+1}, parent={state}] = {F_H1[site, j]:.6f}")


# ==========================================================
# QUESTION 2 PART B — Calculate U = merge(H1,S4)
# ==========================================================
print("\n" + "=" * 80)
print("QUESTION 2 PART B — CALCULATE U = MERGE(H1,S4)")
print("=" * 80)

F_S4 = encoded["S4"]

t_H1_to_U = 0.1
t_S4_to_U = 0.1

P_H1 = expm(Q * t_H1_to_U)
P_S4 = expm(Q * t_S4_to_U)

show("F_H1 used as left child", F_H1)
show("F_S4 used as right child", F_S4)
show(f"P_H1 = P(t={t_H1_to_U})", P_H1)
show(f"P_S4 = P(t={t_S4_to_U})", P_S4)

F_U = felsenstein_merge(F_H1, F_S4, P_H1, P_S4)

show("F_U = likelihood matrix for parent U = merge(H1,S4)", F_U)

print("\nInterpretation:")
print("Rows = sites 1..4")
print("Columns = possible parent states [A, C, G, T]")
print("U is also an internal node represented by a likelihood matrix.")

for site in range(4):
    print(f"\nSite {site+1}: child H1 likelihood + child S4={S['S4'][site]}")
    for j, state in enumerate(DNA):
        print(f"  F_U[site {site+1}, parent={state}] = {F_U[site, j]:.6f}")


# ==========================================================
# FINAL SUMMARY
# ==========================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("""
QUESTION 1:
First merge is selected by:

1. The initial state is:
   S0 = {S1, S2, S3, S4}

2. All possible pairwise merge actions are listed:
   a1 = merge(S1,S2)
   a2 = merge(S1,S3)
   a3 = merge(S1,S4)
   a4 = merge(S2,S3)
   a5 = merge(S2,S4)
   a6 = merge(S3,S4)

3. For each candidate action, the model builds a pairwise representation G_ij.

4. In the actual PhyloGFN architecture, this is conceptually:

   sequence input
   -> seq_emb
   -> Transformer Encoder
   -> pairwise feature construction G_ij
   -> part1_logits_head
   -> logits

5. The real code line is conceptually:

   logits = self.part1_logits_head(x).squeeze(-1)

6. Then softmax converts logits to probabilities:

   P(a_i | S0) = exp(L_i) / sum_j exp(L_j)

7. The action is selected or sampled from this distribution.

Strict limitation:
This standalone script does not load the real trained PhyloGFN model.
Therefore, the neural logits shown here demonstrate the mechanism, but they are not checkpoint-real outputs.
For a full real-code answer, print inside Transformer.forward():

   print(g12.shape)
   print(g12)
   print(logits.shape)
   print(logits)
   print(tree_actions)


QUESTION 2:
After H1 is created, merging H1 with S4 is calculated by Felsenstein recursion:

F_U(site, parent_state)
=
sum over states of H1 branch transition probabilities times F_H1
multiplied by
sum over states of S4 branch transition probabilities times F_S4.

Important:
H1 and U are internal nodes.
They are not DNA letters.
They are likelihood matrices with shape:

(number_of_sites, 4)

For this example:

F_H1 shape = (4, 4)
F_U shape  = (4, 4)
""")

# ==========================================================
# REAL PHYLOGFN SOURCE-CODE PATCHES TO COPY INTO THE REPOSITORY
# ==========================================================
print("\n" + "=" * 80)
print("REAL PHYLOGFN SOURCE-CODE PATCHES")
print("=" * 80)

print("""
COPY THIS INTO Transformer.py, replacing the existing forward_part1 method:

# ------------------------------------------------------------------
# BEGIN PATCH: Transformer.py / forward_part1
# ------------------------------------------------------------------
def forward_part1(self, x, key_padding_mask=None):
    if self.shared_encoder:
        encoder = self.encoder
    else:
        encoder = self.part1_encoder

    x = encoder(x, key_padding_mask)

    summary_token = x[:, :1]
    x = x[:, 1:]
    flow_head_input = summary_token

    if self.concatenate_summary_token:
        _, num_trees, _ = x.shape
        summary_token = summary_token.expand(-1, num_trees, -1)

        x = torch.cat([x, summary_token], dim=2)

        print("\n" + "=" * 80)
        print("REAL PAIRWISE FEATURES x = [candidate_tree_features ; summary_token]")
        print("=" * 80)
        print("Shape:", x.shape)
        print("Meaning: [batch_size, num_candidate_merge_actions, feature_dimension]")
        print("This is the real tensor passed into part1_logits_head.")
        print(x)

    logits = self.part1_logits_head(x).squeeze(-1)

    print("\n" + "=" * 80)
    print("REAL LOGITS FROM part1_logits_head")
    print("=" * 80)
    print("Shape:", logits.shape)
    print("Meaning: one logit score for each candidate merge action")
    print(logits)

    probs = torch.softmax(logits, dim=-1)

    print("\n" + "=" * 80)
    print("REAL ACTION PROBABILITIES AFTER SOFTMAX")
    print("=" * 80)
    print("Shape:", probs.shape)
    print("Meaning: probability distribution over candidate merge actions")
    print(probs)

    if self.compute_state_flow:
        log_state_flow = self.part1_flow_head(flow_head_input)
    else:
        log_state_flow = None

    return logits, log_state_flow
# ------------------------------------------------------------------
# END PATCH: Transformer.py / forward_part1
# ------------------------------------------------------------------
""")

print("""
COPY THIS INTO tb_gfn_phylo.py, inside forward(), replacing the original
lines that compute trees_ret, tree_actions, and tree_pairs:

# ------------------------------------------------------------------
# BEGIN PATCH: tb_gfn_phylo.py / forward
# ------------------------------------------------------------------
trees_ret = self.tree_model(**input_dict)
tree_actions = trees_ret["tree_actions"].cpu().numpy()

print("\n" + "=" * 80)
print("REAL SELECTED ACTION INDICES")
print("=" * 80)
print("Meaning: selected action index for each item in the batch")
print(tree_actions)

tree_pairs = self.env.retrieve_tree_pairs(input_dict["batch_nb_seq"], tree_actions)

print("\n" + "=" * 80)
print("REAL SELECTED TREE PAIRS")
print("=" * 80)
print("Meaning: actual pair of trees selected for merging")
print(tree_pairs)
# ------------------------------------------------------------------
# END PATCH: tb_gfn_phylo.py / forward
# ------------------------------------------------------------------
""")

print("""
STRICT PROFESSOR NOTE:
The standalone demo above is only a numerical explanation.
The two patches above are what you need for the real PhyloGFN trace.
After adding them to the repository and running the real training command,
the terminal will print:

1. real pairwise feature tensor x,
2. real logits from part1_logits_head,
3. real softmax probabilities,
4. real selected action indices,
5. real selected tree pairs.

That is the real evidence your professor asked for.
""")
