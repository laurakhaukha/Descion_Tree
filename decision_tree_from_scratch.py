"""
Decision tree learning pipeline from scratch with NumPy.

Grows a CART tree, prunes it with minimal cost-complexity pruning (alpha chosen
on a validation set), and compares it with an unpruned tree and a random forest.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


# ---------------------------------------------------------------------------
# Impurity measures
# ---------------------------------------------------------------------------
# Both take class probabilities with classes on the last axis.

def gini(p: np.ndarray) -> np.ndarray:
    return 1.0 - (p ** 2).sum(axis=-1)


def entropy(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -(p * logp).sum(axis=-1)


IMPURITY = {"gini": gini, "entropy": entropy}


# ---------------------------------------------------------------------------
# Split search
# ---------------------------------------------------------------------------

def best_split(X, y, n_classes, impurity, min_samples_leaf, features):
    """Best (feature, threshold, gain) over the given features, or None.

    For each feature the samples are sorted once, and cumulative class counts
    give the left/right class distribution of every candidate threshold at once,
    so a feature costs O(n log n) instead of O(n^2).
    """
    n = len(y)
    total = np.bincount(y, minlength=n_classes).astype(float)
    parent = impurity(total / n)
    n_left = np.arange(1, n, dtype=float)
    n_right = n - n_left
    one_hot = np.eye(n_classes)

    best, best_gain = None, -np.inf
    for f in features:
        order = np.argsort(X[:, f], kind="stable")
        xs, ys = X[order, f], y[order]
        left = np.cumsum(one_hot[ys], axis=0)[:-1]  # counts left of each gap
        right = total - left
        child = (n_left * impurity(left / n_left[:, None])
                 + n_right * impurity(right / n_right[:, None])) / n
        # A threshold can only sit between two *different* values.
        valid = (xs[:-1] < xs[1:]) & (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not valid.any():
            continue
        gains = np.where(valid, parent - child, -np.inf)
        i = int(np.argmax(gains))
        if gains[i] > best_gain:
            threshold = 0.5 * (xs[i] + xs[i + 1])
            if threshold == xs[i + 1]:  # midpoint rounded up between adjacent floats
                threshold = xs[i]
            best, best_gain = (int(f), float(threshold), float(gains[i])), gains[i]
    return best


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

class DecisionTree:
    """CART classifier stored as flat arrays (node i's children are left[i], right[i]).

    Nodes are created parent-first, so every child has a larger index than its
    parent. Leaves have feature == -1. `value` holds the training class counts.
    """

    def __init__(self, criterion: str = "gini", max_depth: int | None = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 min_impurity_decrease: float = 0.0, max_features: int | None = None,
                 rng: np.random.Generator | None = None):
        if criterion not in IMPURITY:
            raise ValueError(f"unknown criterion '{criterion}'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.min_samples_leaf = max(1, min_samples_leaf)
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()

    # -- training -----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> "DecisionTree":
        impurity = IMPURITY[self.criterion]
        self.n_classes = n_classes or int(y.max()) + 1
        self.n_features = X.shape[1]
        feature, threshold, left, right, value, imp, n_samples = [], [], [], [], [], [], []

        # Explicit stack instead of recursion, so deep trees can't hit Python's limit.
        stack = [(np.arange(len(y)), 0, -1, False)]  # (sample indices, depth, parent, is_left)
        while stack:
            idx, depth, parent, is_left = stack.pop()
            node = len(feature)
            if parent >= 0:
                (left if is_left else right)[parent] = node

            counts = np.bincount(y[idx], minlength=self.n_classes).astype(float)
            node_imp = float(impurity(counts / len(idx)))
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(counts)
            imp.append(node_imp)
            n_samples.append(len(idx))

            if (node_imp == 0.0 or len(idx) < self.min_samples_split
                    or (self.max_depth is not None and depth >= self.max_depth)):
                continue
            features = np.arange(self.n_features)
            if self.max_features is not None and self.max_features < self.n_features:
                features = self.rng.choice(self.n_features, self.max_features, replace=False)
            split = best_split(X[idx], y[idx], self.n_classes, impurity, self.min_samples_leaf, features)
            # Gain is weighted by the node's share of the data, as in scikit-learn.
            if split is None or split[2] * len(idx) / len(y) + 1e-12 < self.min_impurity_decrease:
                continue

            f, t, _ = split
            feature[node], threshold[node] = f, t
            go_left = X[idx, f] <= t
            stack.append((idx[~go_left], depth + 1, node, False))
            stack.append((idx[go_left], depth + 1, node, True))

        self.feature = np.array(feature)
        self.threshold = np.array(threshold)
        self.left = np.array(left)
        self.right = np.array(right)
        self.value = np.array(value)
        self.impurity = np.array(imp)
        self.n_samples = np.array(n_samples)
        return self

    # -- inference ----------------------------------------------------------

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Leaf index for every sample, routing all samples down one level at a time."""
        node = np.zeros(len(X), dtype=int)
        active = self.feature[node] >= 0
        while active.any():
            n = node[active]
            go_left = X[active, self.feature[n]] <= self.threshold[n]
            node[active] = np.where(go_left, self.left[n], self.right[n])
            active = self.feature[node] >= 0
        return node

    def predict_proba(self, X):
        counts = self.value[self.apply(X)]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # -- structure ----------------------------------------------------------

    def _reachable(self) -> np.ndarray:
        """Pruning turns nodes into leaves but leaves their old subtrees in the arrays."""
        reach = np.zeros(len(self.feature), dtype=bool)
        reach[0] = True
        for t in range(len(self.feature)):
            if reach[t] and self.feature[t] >= 0:
                reach[self.left[t]] = reach[self.right[t]] = True
        return reach

    @property
    def n_leaves(self) -> int:
        return int((self._reachable() & (self.feature < 0)).sum())

    @property
    def depth(self) -> int:
        depth = np.zeros(len(self.feature), dtype=int)
        for t in range(len(self.feature)):
            if self.feature[t] >= 0:
                depth[self.left[t]] = depth[self.right[t]] = depth[t] + 1
        return int(depth[self._reachable()].max())

    def feature_importances(self) -> np.ndarray:
        """Total weighted impurity decrease per feature, normalised to sum to 1."""
        w = self.n_samples / self.n_samples[0] * self.impurity
        imp = np.zeros(self.n_features)
        for t in np.flatnonzero(self._reachable() & (self.feature >= 0)):
            imp[self.feature[t]] += w[t] - w[self.left[t]] - w[self.right[t]]
        return imp / imp.sum() if imp.sum() > 0 else imp

    def copy(self) -> "DecisionTree":
        tree = DecisionTree.__new__(DecisionTree)
        tree.__dict__.update({k: v.copy() if isinstance(v, np.ndarray) else v
                              for k, v in self.__dict__.items()})
        return tree

    # -- minimal cost-complexity pruning ------------------------------------
    # R(T) = sum over leaves of (n_leaf / N) * impurity(leaf). Collapsing node t
    # costs R(t) - R(T_t) in training impurity and saves |leaves(T_t)| - 1 leaves,
    # so g(t) = that ratio is the alpha at which pruning t becomes worthwhile.

    def _weakest_links(self):
        r_node = self.n_samples / self.n_samples[0] * self.impurity
        r_sub = r_node.copy()
        leaves = np.ones(len(self.feature))
        for t in reversed(range(len(self.feature))):  # children before parents
            if self.feature[t] >= 0:
                r_sub[t] = r_sub[self.left[t]] + r_sub[self.right[t]]
                leaves[t] = leaves[self.left[t]] + leaves[self.right[t]]
        internal = np.flatnonzero(self._reachable() & (self.feature >= 0))
        g = (r_node[internal] - r_sub[internal]) / (leaves[internal] - 1)
        g_min = g.min()
        return float(g_min), internal[g <= g_min + 1e-12]

    def _collapse(self, nodes):
        self.feature[nodes] = -1
        self.left[nodes] = self.right[nodes] = -1

    def cost_complexity_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Effective alphas at which the tree shrinks, and the leaf count after each."""
        tree = self.copy()
        alphas, leaves = [0.0], [tree.n_leaves]
        while tree.n_leaves > 1:
            g, nodes = tree._weakest_links()
            tree._collapse(nodes)
            alphas.append(max(g, alphas[-1]))
            leaves.append(tree.n_leaves)
        return np.array(alphas), np.array(leaves)

    def pruned(self, alpha: float) -> "DecisionTree":
        """Smallest subtree minimising R(T) + alpha * |leaves(T)|."""
        tree = self.copy()
        while tree.n_leaves > 1:
            g, nodes = tree._weakest_links()
            if g > alpha + 1e-12:
                break
            tree._collapse(nodes)
        return tree

    # -- export -------------------------------------------------------------

    def export_text(self, feature_names=None, class_names=None, max_depth: int = 3) -> str:
        feature_names = feature_names or [f"x{i}" for i in range(self.n_features)]
        class_names = class_names or [str(c) for c in range(self.n_classes)]
        lines = []

        def walk(t, depth):
            pad = "|   " * depth
            if self.feature[t] < 0 or depth >= max_depth:
                counts = self.value[t].astype(int).tolist()
                more = "" if self.feature[t] < 0 else " (truncated)"
                lines.append(f"{pad}|--- class {class_names[int(np.argmax(self.value[t]))]} "
                             f"{counts}{more}")
                return
            name, thr = feature_names[self.feature[t]], self.threshold[t]
            lines.append(f"{pad}|--- {name} <= {thr:.3f}")
            walk(self.left[t], depth + 1)
            lines.append(f"{pad}|--- {name} >  {thr:.3f}")
            walk(self.right[t], depth + 1)

        walk(0, 0)
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save the reachable tree as JSON (nested dicts are easy to read and diff)."""
        def node(t):
            if self.feature[t] < 0:
                return {"value": self.value[t].astype(int).tolist()}
            return {"feature": int(self.feature[t]), "threshold": float(self.threshold[t]),
                    "left": node(self.left[t]), "right": node(self.right[t])}

        with open(path, "w") as fh:
            json.dump({"criterion": self.criterion, "n_classes": self.n_classes,
                       "n_features": self.n_features, "root": node(0)}, fh)

    @classmethod
    def load(cls, path: str) -> "DecisionTree":
        """Rebuild a tree for prediction (pruning needs the training statistics, which aren't saved)."""
        with open(path) as fh:
            data = json.load(fh)
        tree = cls(criterion=data["criterion"])
        tree.n_classes, tree.n_features = data["n_classes"], data["n_features"]
        feature, threshold, left, right, value = [], [], [], [], []
        stack = [(data["root"], -1, False)]
        while stack:
            d, parent, is_left = stack.pop()
            t = len(feature)
            if parent >= 0:
                (left if is_left else right)[parent] = t
            leaf = "value" in d
            feature.append(-1 if leaf else d["feature"])
            threshold.append(0.0 if leaf else d["threshold"])
            left.append(-1)
            right.append(-1)
            value.append(d["value"] if leaf else [0] * tree.n_classes)
            if not leaf:
                stack.append((d["right"], t, False))
                stack.append((d["left"], t, True))
        tree.feature, tree.threshold = np.array(feature), np.array(threshold)
        tree.left, tree.right = np.array(left), np.array(right)
        tree.value = np.array(value, dtype=float)
        return tree


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------

class RandomForest:
    """Bagged trees that each consider a random feature subset at every split."""

    def __init__(self, n_trees: int = 100, max_features: int | None = None, rng=None, **tree_kwargs):
        self.n_trees = n_trees
        self.max_features = max_features
        self.rng = rng or np.random.default_rng()
        self.tree_kwargs = tree_kwargs

    def fit(self, X, y, n_classes=None):
        n, d = X.shape
        self.n_classes = n_classes or int(y.max()) + 1
        m = self.max_features or max(1, int(np.sqrt(d)))
        self.trees = []
        oob_proba = np.zeros((n, self.n_classes))
        for _ in range(self.n_trees):
            boot = self.rng.integers(0, n, n)
            tree = DecisionTree(max_features=m, rng=self.rng, **self.tree_kwargs)
            self.trees.append(tree.fit(X[boot], y[boot], self.n_classes))
            # Roughly 37% of samples are left out of each bootstrap: a free validation set.
            oob = np.ones(n, dtype=bool)
            oob[boot] = False
            oob_proba[oob] += tree.predict_proba(X[oob])
        seen = oob_proba.sum(axis=1) > 0
        self.oob_score = float((oob_proba[seen].argmax(axis=1) == y[seen]).mean())
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def feature_importances(self):
        return np.mean([t.feature_importances() for t in self.trees], axis=0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_spiral(n_per_class: int, n_classes: int, noise: float, rng: np.random.Generator):
    """Interleaved 2-D spirals: curved boundaries that axis-aligned splits must approximate."""
    n = n_per_class
    X = np.zeros((n * n_classes, 2))
    y = np.zeros(n * n_classes, dtype=int)
    for c in range(n_classes):
        idx = slice(c * n, (c + 1) * n)
        radius = np.linspace(0.0, 1.0, n)
        theta = np.linspace(c * 4.0, (c + 1) * 4.0, n) + rng.standard_normal(n) * noise
        X[idx] = np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])
        y[idx] = c
    return X, y


def train_val_test_split(X, y, rng, val_frac=0.15, test_frac=0.15):
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    n_val = int(len(X) * val_frac)
    test, val, train = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return (X[train], y[train]), (X[val], y[val]), (X[test], y[test])


# ---------------------------------------------------------------------------
# Checks, evaluation and plotting
# ---------------------------------------------------------------------------

def split_check(rng, n=40, d=3, n_classes=3) -> float:
    """Compare the vectorised split search with a brute-force loop over every threshold.

    Returns the largest gap between the best gains; ~0 means best_split is correct.
    Values are rounded so that tied feature values are exercised too.
    """
    X = rng.standard_normal((n, d)).round(1)
    y = rng.integers(0, n_classes, n)
    worst = 0.0
    for name, impurity in IMPURITY.items():
        for min_leaf in (1, 5):
            fast = best_split(X, y, n_classes, impurity, min_leaf, range(d))
            parent = impurity(np.bincount(y, minlength=n_classes) / n)
            brute = -np.inf
            for f in range(d):
                values = np.unique(X[:, f])
                for t in (values[:-1] + values[1:]) / 2:
                    mask = X[:, f] <= t
                    nl, nr = mask.sum(), (~mask).sum()
                    if nl < min_leaf or nr < min_leaf:
                        continue
                    pl = np.bincount(y[mask], minlength=n_classes) / nl
                    pr = np.bincount(y[~mask], minlength=n_classes) / nr
                    brute = max(brute, parent - (nl * impurity(pl) + nr * impurity(pr)) / n)
            fast_gain = fast[2] if fast else -np.inf
            worst = max(worst, 0.0 if fast_gain == brute else abs(fast_gain - brute))
    return worst


def accuracy(model, X, y) -> float:
    return float((model.predict(X) == y).mean())


def plot_results(models, X, y, alphas, leaves, train_acc, val_acc, best_alpha, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot (pip install matplotlib).")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    # Decision boundaries on the first two features (extra features held at 0).
    pad = 0.3
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 300),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 300),
    )
    grid = np.zeros((xx.size, X.shape[1]))
    grid[:, 0], grid[:, 1] = xx.ravel(), yy.ravel()
    for ax, (name, model) in zip(axes.ravel()[:3], models.items()):
        ax.contourf(xx, yy, model.predict(grid).reshape(xx.shape), alpha=0.3, cmap="viridis")
        ax.scatter(X[:, 0], X[:, 1], c=y, s=12, cmap="viridis", edgecolors="k", linewidths=0.3)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    # Pruning path: accuracy as the tree shrinks.
    ax = axes[1, 1]
    ax.plot(leaves, train_acc, marker=".", label="train")
    ax.plot(leaves, val_acc, marker=".", label="validation")
    best = int(np.flatnonzero(alphas == best_alpha)[0])
    ax.axvline(leaves[best], color="gray", ls="--", label=f"chosen ({leaves[best]} leaves)")
    ax.set_xscale("log")
    ax.set_xlabel("number of leaves (pruning path)")
    ax.set_ylabel("accuracy")
    ax.set_title("Cost-complexity pruning")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Decision tree learning pipeline from scratch (NumPy).")
    ap.add_argument("--criterion", choices=list(IMPURITY), default="gini")
    ap.add_argument("--max-depth", type=int, default=None, help="limit for the grown tree (default: none)")
    ap.add_argument("--min-samples-leaf", type=int, default=1)
    ap.add_argument("--trees", type=int, default=100, help="random forest size (0 to skip)")
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--samples", type=int, default=300, help="samples per class")
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--noise-features", type=int, default=0,
                    help="append pure-noise features (watch their importance stay near 0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true", help="save decision boundaries + pruning curve")
    ap.add_argument("--save", type=str, default=None, help="path to save the pruned tree (.json)")
    ap.add_argument("--skip-check", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 1. Verify the vectorised split search against brute force.
    if not args.skip_check:
        err = split_check(rng)
        status = "OK" if err < 1e-10 else "CHECK best_split"
        print(f"Split check: max gain difference vs brute force = {err:.2e}  [{status}]\n")

    # 2. Data. Trees only compare values within a feature, so no standardisation needed.
    X, y = make_spiral(args.samples, args.classes, args.noise, rng)
    if args.noise_features:
        X = np.hstack([X, rng.uniform(X.min(), X.max(), (len(X), args.noise_features))])
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = train_val_test_split(X, y, rng)
    names = ["x0", "x1"] + [f"noise{i}" for i in range(args.noise_features)]
    print(f"Data: train {len(X_tr)}, val {len(X_va)}, test {len(X_te)} | "
          f"features {X.shape[1]} | classes {args.classes}")

    # 3. Grow a tree until its leaves are pure (or a stopping rule hits).
    start = time.perf_counter()
    full = DecisionTree(args.criterion, max_depth=args.max_depth,
                        min_samples_leaf=args.min_samples_leaf).fit(X_tr, y_tr, args.classes)
    print(f"Grown tree: depth {full.depth}, {full.n_leaves} leaves "
          f"({time.perf_counter() - start:.2f}s)")

    # 4. Prune: walk the cost-complexity path and keep the alpha that scores best
    #    on validation data. Ties go to the larger alpha, i.e. the simpler tree.
    alphas, leaves = full.cost_complexity_path()
    path_trees = [full.pruned(a) for a in alphas]
    train_acc = np.array([accuracy(t, X_tr, y_tr) for t in path_trees])
    val_acc = np.array([accuracy(t, X_va, y_va) for t in path_trees])
    best = int(np.flatnonzero(val_acc == val_acc.max())[-1])
    pruned = path_trees[best]
    print(f"Pruning path: {len(alphas)} subtrees | chosen alpha {alphas[best]:.5f} -> "
          f"depth {pruned.depth}, {pruned.n_leaves} leaves, val acc {val_acc[best]:.3f}")

    # 5. Random forest for comparison.
    models = {"Full tree": full, "Pruned tree": pruned}
    if args.trees:
        start = time.perf_counter()
        forest = RandomForest(args.trees, rng=rng, criterion=args.criterion,
                              min_samples_leaf=args.min_samples_leaf)
        forest.fit(X_tr, y_tr, args.classes)
        models[f"Random forest ({args.trees} trees)"] = forest
        print(f"Random forest: {args.trees} trees, OOB accuracy {forest.oob_score:.3f} "
              f"({time.perf_counter() - start:.2f}s)")

    # 6. Final evaluation on held-out test data.
    print(f"\n{'model':<28}{'leaves':>8}{'train':>8}{'val':>8}{'test':>8}")
    for name, model in models.items():
        n_leaves = (model.n_leaves if isinstance(model, DecisionTree)
                    else np.mean([t.n_leaves for t in model.trees]))
        print(f"{name:<28}{n_leaves:>8.0f}{accuracy(model, X_tr, y_tr):>8.3f}"
              f"{accuracy(model, X_va, y_va):>8.3f}{accuracy(model, X_te, y_te):>8.3f}")

    print("\nFeature importances (pruned tree):")
    for name, imp in zip(names, pruned.feature_importances()):
        print(f"  {name:<8} {imp:.3f}")
    print("\nPruned tree (top 3 levels):")
    print(pruned.export_text(names, max_depth=3))

    if args.save:
        pruned.save(args.save)
        print(f"\nSaved pruned tree to {args.save}")

    if args.plot:
        X_all = np.vstack([X_tr, X_va, X_te])
        y_all = np.concatenate([y_tr, y_va, y_te])
        plot_results(models, X_all, y_all, alphas, leaves, train_acc, val_acc,
                     alphas[best], "decision_tree_results.png")


if __name__ == "__main__":
    main()
