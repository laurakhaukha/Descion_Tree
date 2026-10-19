# Decision Tree Learning Pipeline from Scratch (NumPy)

A complete decision tree pipeline written with nothing but NumPy: no scikit-learn. It grows a CART tree, prunes it with cost-complexity pruning, compares it with a random forest, and explains what it learned. Every step is plain, readable code.

![Decision boundaries and pruning path](decision_tree_results.png)

## Pipeline

1. **Check:** tests the vectorised split search against a brute-force loop over every threshold.
2. **Grow:** builds a CART tree (Gini or entropy) until its leaves are pure or a stopping rule is hit.
3. **Prune:** computes the minimal cost-complexity pruning path and picks the alpha that gives the best validation accuracy.
4. **Compare:** trains a random forest (bootstrap + random feature subsets, with out-of-bag accuracy).
5. **Report:** prints test accuracy, feature importances and the tree's rules, and optionally saves the tree as JSON and a plot.

## Quick start

```bash
pip install -r requirements.txt
python decision_tree_from_scratch.py --plot
```

Example output:

```
Split check: max gain difference vs brute force = 0.00e+00  [OK]

Data: train 630, val 135, test 135 | features 2 | classes 3
Grown tree: depth 8, 19 leaves (0.01s)
Pruning path: 14 subtrees | chosen alpha 0.00544 -> depth 6, 11 leaves, val acc 0.978
Random forest: 100 trees, OOB accuracy 0.986 (0.56s)

model                         leaves   train     val    test
Full tree                         19   0.997   0.970   0.985
Pruned tree                       11   0.986   0.978   0.978
Random forest (100 trees)         29   0.997   0.978   0.985

Feature importances (pruned tree):
  x0       0.504
  x1       0.496

Pruned tree (top 3 levels):
|--- x1 <= -0.064
|   |--- x1 <= -0.496
|   |   |--- class 0 [74, 0, 1]
...
```

Pruning cuts the tree from 19 leaves to 11 and loses very little test accuracy.

## What's inside

| Component | Details |
|---|---|
| Splitting | Sort once per feature; cumulative class counts score every threshold in one vectorised pass |
| Criteria | Gini impurity, entropy |
| Stopping rules | `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease` |
| Pruning | Minimal cost-complexity (weakest-link) pruning, alpha chosen on validation data |
| Ensemble | Random forest with out-of-bag score |
| Interpretability | Feature importances, text export of the decision rules |
| Extras | JSON save / load, decision-boundary and pruning-curve plot |

## Options

```bash
python decision_tree_from_scratch.py --criterion entropy --min-samples-leaf 3
python decision_tree_from_scratch.py --max-depth 5 --classes 4 --noise 0.3
python decision_tree_from_scratch.py --noise-features 3      # importance of noise features stays ~0
python decision_tree_from_scratch.py --trees 0 --save tree.json
python decision_tree_from_scratch.py --help
```

## How it works

**Split search.** For each feature, samples are sorted once. A cumulative sum of one-hot labels gives the class counts on the left of every possible threshold, and the right side is the total minus the left. The impurity of every candidate split is then computed in one vectorised step, so each feature costs O(n log n) instead of O(n²).

**Tree storage.** The tree is stored as flat arrays (`feature`, `threshold`, `left`, `right`, `value`). Prediction sends every sample down one level at a time, rather than looping over samples in Python.

**Pruning.** A fully grown tree overfits. Cost-complexity pruning scores a tree by `R(T) + alpha * |leaves|`, where `R(T)` is the weighted impurity of its leaves. It repeatedly collapses the node with the smallest

```
g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)
```

which is the training impurity added per leaf removed. It repeats this until only the root is left, and each step gives one candidate tree. The candidate that scores best on the validation set wins, and ties go to the smaller tree.

**Random forest.** Each tree is trained on a bootstrap sample and considers only a random subset of features at each split. Roughly 37% of samples are left out of each bootstrap, so they give a free out-of-bag estimate of accuracy.

## Verification

- The built-in split check compares the fast split search with brute force on random data containing tied values.
- Cross-checked against scikit-learn's `DecisionTreeClassifier` on the same data, with both Gini and entropy. The trees have the same number of leaves and agree on 100% of predictions. The pruning steps and the pruned tree sizes match too.

## Requirements

- Python 3.9+
- NumPy
- matplotlib (optional, only for `--plot`)
