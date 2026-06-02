import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm

# I keep the numbers short in the terminal output.
np.set_printoptions(precision=4, suppress=True)
torch.manual_seed(42)

DNA_STATES = ["A", "C", "G", "T"]

ONE_HOT = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "N": [1.0, 1.0, 1.0, 1.0],
}


def title(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def show(name, value):
    print(f"\n{name}")
    print(f"shape = {value.shape}")
    print(value)


def seq_to_array(seq):
    """Convert a DNA string to one-hot matrix."""
    return np.array([ONE_HOT[ch] for ch in seq], dtype=float)


def jukes_cantor_Q():
    """JC rate matrix for states A,C,G,T."""
    Q = np.ones((4, 4), dtype=float) / 3.0
    np.fill_diagonal(Q, -1.0)
    return Q


def merge_likelihood(F_left, F_right, P_left, P_right):
    """
    Felsenstein recursion for one internal node.

    For each site and each possible parent state:
    likelihood = left contribution * right contribution
    """
    n_sites = F_left.shape[0]
    F_parent = np.zeros((n_sites, 4), dtype=float)

    for site in range(n_sites):
        for parent_state in range(4):
            left_term = np.sum(P_left[parent_state, :] * F_left[site, :])
            right_term = np.sum(P_right[parent_state, :] * F_right[site, :])
            F_parent[site, parent_state] = left_term * right_term

    return F_parent


# ---------------------------------------------------------------------
# Input example used in the meeting notes
# ---------------------------------------------------------------------
sequences = {
    "S1": "AATG",
    "S2": "AATT",
    "S3": "ATGG",
    "S4": "ACGC",
}

title("STEP 0 — INPUT SEQUENCES")
for name, seq in sequences.items():
    print(f"{name} = {seq}")


# ---------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------
title("STEP 1 — ONE-HOT ENCODING")

encoded = {}
for name, seq in sequences.items():
    encoded[name] = seq_to_array(seq)
    show(f"F_{name}", encoded[name])

print("\nRows are sites. Columns are [A, C, G, T].")
print("Here each sequence has 4 sites, so each matrix is (4, 4).")


# ---------------------------------------------------------------------
# First merge action
# ---------------------------------------------------------------------
title("QUESTION 1 — HOW THE FIRST MERGE ACTION IS SELECTED")

current_state = list(sequences.keys())
candidate_actions = list(itertools.combinations(current_state, 2))

print(f"Current state: {current_state}")
print(f"Number of possible first merges: {len(candidate_actions)}")

for idx, pair in enumerate(candidate_actions, start=1):
    print(f"a{idx} = merge{pair}")


# ---------------------------------------------------------------------
# Simple raw G_ij examples
# ---------------------------------------------------------------------
title("STEP 2 — SIMPLE G_ij EXAMPLES FROM ONE-HOT DATA")

print("These are only simple raw concatenations to understand dimensions.")
print("They are not the real internal PhyloGFN representations.")

for idx, (left, right) in enumerate(candidate_actions, start=1):
    flat = np.concatenate([encoded[left].flatten(), encoded[right].flatten()])
    rows = np.concatenate([encoded[left], encoded[right]], axis=0)
    cols = np.concatenate([encoded[left], encoded[right]], axis=1)

    print(f"\nAction a{idx}: merge({left}, {right})")
    show(f"G_{left}{right}_flat", flat)
    show(f"G_{left}{right}_rows", rows)
    show(f"G_{left}{right}_cols", cols)


# ---------------------------------------------------------------------
# Small neural-network style demonstration
# ---------------------------------------------------------------------
title("STEP 3 — LOGITS AND SOFTMAX DEMONSTRATION")

print("""
This part is only a demonstration of the mechanism.

It does not load a trained PhyloGFN checkpoint.
The tree embeddings and summary token below are simulated.
The real values must be printed from the actual Transformer.forward_part1().
""")

embedding_size = 128
num_trees = 4

tree_embeddings = torch.randn(num_trees, embedding_size)
summary_token = torch.randn(embedding_size)

pair_indices = list(itertools.combinations(range(num_trees), 2))

pair_vectors = []
for i, j in pair_indices:
    pair_representation = torch.cat(
        [summary_token, tree_embeddings[i] + tree_embeddings[j]],
        dim=0,
    )
    pair_vectors.append(pair_representation)

pair_batch = torch.stack(pair_vectors)

print("Simulated tree embeddings:", tree_embeddings.shape)
print("Simulated summary token:", summary_token.shape)
print("Pair representation batch:", pair_batch.shape)
print("Meaning: 6 candidate actions, each represented by 256 numbers.")

mlp_head = nn.Sequential(
    nn.Linear(embedding_size * 2, 256),
    nn.GELU(),
    nn.Linear(256, 1),
)

with torch.no_grad():
    logits = mlp_head(pair_batch).squeeze(-1)

print("\nLogits from the demonstration MLP:")
for k, (i, j) in enumerate(pair_indices):
    print(f"L_{i + 1}{j + 1} = {logits[k].item():.6f}")

probs = F.softmax(logits, dim=0)

print("\nSoftmax probabilities:")
for k, (i, j) in enumerate(pair_indices):
    print(f"P(merge S{i + 1}, S{j + 1}) = {probs[k].item():.6f}")

chosen_idx = torch.argmax(probs).item()
chosen_pair = pair_indices[chosen_idx]

print("\nChosen action in this demonstration:")
print(f"a{chosen_idx + 1} = merge(S{chosen_pair[0] + 1}, S{chosen_pair[1] + 1})")


# ---------------------------------------------------------------------
# H1 = merge(S1, S2)
# ---------------------------------------------------------------------
title("QUESTION 2A — H1 = MERGE(S1, S2)")

Q = jukes_cantor_Q()
show("Jukes-Cantor Q", Q)
print("Row sums:", Q.sum(axis=1))

t_left = 0.1
t_right = 0.1

P_left = expm(Q * t_left)
P_right = expm(Q * t_right)

show(f"P_left, t={t_left}", P_left)
show(f"P_right, t={t_right}", P_right)

F_H1 = merge_likelihood(
    encoded["S1"],
    encoded["S2"],
    P_left,
    P_right,
)

show("F_H1", F_H1)

print("\nH1 is not a nucleotide. It is an internal-node likelihood matrix.")
print("Rows are sites; columns are possible parent states [A, C, G, T].")

for site in range(4):
    print(f"\nSite {site + 1}: S1={sequences['S1'][site]}, S2={sequences['S2'][site]}")
    for state_idx, state in enumerate(DNA_STATES):
        print(f"  F_H1[{site + 1}, {state}] = {F_H1[site, state_idx]:.6f}")


# ---------------------------------------------------------------------
# U = merge(H1, S4)
# ---------------------------------------------------------------------
title("QUESTION 2B — U = MERGE(H1, S4)")

F_S4 = encoded["S4"]

P_H1 = expm(Q * 0.1)
P_S4 = expm(Q * 0.1)

show("F_H1 used as left child", F_H1)
show("F_S4 used as right child", F_S4)

F_U = merge_likelihood(
    F_H1,
    F_S4,
    P_H1,
    P_S4,
)

show("F_U", F_U)

print("\nU is also an internal-node likelihood matrix.")

for site in range(4):
    print(f"\nSite {site + 1}: H1 + S4={sequences['S4'][site]}")
    for state_idx, state in enumerate(DNA_STATES):
        print(f"  F_U[{site + 1}, {state}] = {F_U[site, state_idx]:.6f}")


# ---------------------------------------------------------------------
# What needs to be printed from the real PhyloGFN code
# ---------------------------------------------------------------------
title("REAL PHYLOGFN TRACE NEEDED IN THE REPOSITORY")

print("""
This script is only a controlled example.

For the real PhyloGFN model, the following must be printed from the actual
source code during train.py:

1. pair representation tensor x
   expected shape: [batch_size, num_candidate_actions, feature_dim]

2. logits from part1_logits_head
   expected shape: [batch_size, num_candidate_actions]

3. softmax probabilities
   expected shape: [batch_size, num_candidate_actions]

4. selected action indices
   expected shape: [batch_size]

5. selected tree pairs

For the four-leaf example, if batch_size = 64 and embedding_size = 128,
the expected shapes are:

pair representations: torch.Size([64, 6, 256])
logits:               torch.Size([64, 6])
probabilities:        torch.Size([64, 6])
selected actions:     (64,)
""")
