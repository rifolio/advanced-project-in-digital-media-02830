"""Embedding audit for reveal data: does the tutor reply cluster by things that should not matter?

  uv sync --extra embed
  uv run embed_audit.py --model minilm                       # one embedder (see config.EMBEDDERS)
  uv run embed_audit.py --model gemini-embedding-001 & uv run embed_audit.py --model minilm & wait
  uv run embed_audit.py --table                              # markdown table from results/embed-audit/*.json
  uv run embed_audit.py --selftest

Embeds `response` of reveal_v2, the v1 gsm8k file and MRBench (cached in runs/embed/, git-ignored) and runs
linear probes (logistic regression, grouped CV by problem), label transfer to MRBench, silhouette,
domain gap, minimal-pair distances and 2D plots. MRBench is only ever an evaluation target.
Words-only counterpart: shortcut_audit.py.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

import config

V2, V1, MRB = "data/reveal_v2.jsonl", "data/gsm8k.jsonl", "data/mrbench.jsonl"
CACHE_DIR, OUT = Path("runs/embed"), Path("results/embed-audit")
BATCH = 100  # Gemini batch limit per request


def load(path):
    return [json.loads(line) for line in open(path)]


# ---------- embedding ----------

def key(text):
    return hashlib.sha1(text.encode()).hexdigest()


def embed_texts(name, texts):
    """Unit-length embeddings, one row per text. Only texts missing from runs/embed/<name>.npz are embedded."""
    spec = config.EMBEDDERS[name]
    path = CACHE_DIR / f"{name}.npz"
    cache = {}
    if path.exists():
        z = np.load(path)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = sorted({t for t in texts if key(t) not in cache})
    tokens = 0
    if todo and spec["kind"] == "st":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(spec["model"])
        vecs = model.encode(todo, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        cache.update({key(t): v for t, v in zip(todo, vecs)})
    elif todo:
        import litellm

        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            r = litellm.embedding(model=spec["model"], input=chunk, dimensions=config.EMBED_DIMS,
                                  num_retries=config.NUM_RETRIES)
            tokens += r.usage.prompt_tokens
            for t, d in zip(chunk, r.data):
                v = np.array(d["embedding"], dtype=np.float32)
                cache[key(t)] = v / np.linalg.norm(v)
            print(f"  {name}: {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
    if todo:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())).astype(np.float32))
    if tokens:
        print(f"  {name}: {len(todo)} new texts, {tokens} prompt tokens (as reported by the API)")
    return np.stack([cache[key(t)] for t in texts]).astype(np.float32), tokens


# ---------- metrics ----------

def probe_cv(X, y, groups, folds=5):
    """Out-of-fold AUC of a logistic-regression probe, folds grouped so no group is in train and test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))
        oof[te] = m.fit(X[tr], y[tr]).decision_function(X[te])
    return float(roc_auc_score(y, oof))


def fit_probe(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced")).fit(Xtr, ytr)


def transfer_auc(model, X, y):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, model.decision_function(X)))


def silhouette(X, labels):
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(X, labels, metric="cosine"))


def nn_cosine(A, B, a_groups=None, b_groups=None):
    """Mean over rows of A of the highest cosine to any row of B; rows of B in the same group are skipped."""
    S = A @ B.T
    if a_groups is not None:
        S[np.asarray(a_groups)[:, None] == np.asarray(b_groups)[None, :]] = -1
    return float(S.max(axis=1).mean())


def pair_cosines(rows, X, seed=0):
    """Guide-reveal cosine of the same context, split by minimal pair vs free, and vs a random reveal as a floor."""
    ctx = {}
    for i, r in enumerate(rows):
        ctx.setdefault(r["id"].rsplit("_", 1)[0], {})[r["label"]] = i
    out, rng = {}, np.random.default_rng(seed)
    for kind in ("pair", "free"):
        ids = [c for c, d in ctx.items() if len(d) == 2 and bool(rows[d["yes"]].get("pair_id")) == (kind == "pair")]
        g = np.array([ctx[c]["no"] for c in ids])
        v = np.array([ctx[c]["yes"] for c in ids])
        out[kind] = float((X[g] * X[v]).sum(1).mean())
        out[kind + "_n"] = len(ids)
        out[kind + "_random"] = float((X[g] * X[rng.permutation(v)]).sum(1).mean())
    return out


# ---------- audit ----------

def audit(name, plots=True):
    v2, v1, mrb = load(V2), load(V1), load(MRB)
    texts = [r["response"] for r in v2 + v1 + mrb]
    E, tokens = embed_texts(name, texts)
    X2, X1, XM = E[:len(v2)], E[len(v2):len(v2) + len(v1)], E[len(v2) + len(v1):]
    y = lambda rows: np.array([r["label"] == "yes" for r in rows])
    res = {"embedder": name, "model": config.EMBEDDERS[name]["model"], "dim": int(E.shape[1]), "tokens": tokens}

    # 1. probes inside reveal_v2, grouped by problem
    grp = np.array([r["problem_id"] for r in v2])
    gen = np.array([r["generator"] for r in v2])
    two = np.isin(gen, ["claude-haiku-5-5", "gemini"])
    isclaude = (gen == "claude-haiku-5-5")
    src = np.array([r["context_source"] == "gsm8k_synthetic" for r in v2])
    lab = y(v2)
    p = {"label": probe_cv(X2, lab, grp), "generator": probe_cv(X2[two], isclaude[two], grp[two]),
         "context_source": probe_cv(X2, src, grp),
         "generator_in_yes": probe_cv(X2[two & lab], isclaude[two & lab], grp[two & lab]),
         "generator_in_no": probe_cv(X2[two & ~lab], isclaude[two & ~lab], grp[two & ~lab])}
    res["probe_cv"] = p

    # 2. label transfer
    split = np.array([r["split"] for r in v2])
    ym, srcm = y(mrb), np.array([r["source"] for r in mrb])
    t = {}
    m2 = fit_probe(X2[split == "train"], lab[split == "train"])
    t["v2_val"] = transfer_auc(m2, X2[split == "val"], lab[split == "val"])
    sp1 = np.array([r["split"] for r in v1])
    m1 = fit_probe(X1[sp1 == "train"], y(v1)[sp1 == "train"])
    t["v1_val"] = transfer_auc(m1, X1[sp1 == "val"], y(v1)[sp1 == "val"])
    for tag, m in (("v2", m2), ("v1", m1)):
        t[f"{tag}_mrbench"] = transfer_auc(m, XM, ym)
        for s in sorted(set(srcm)):
            t[f"{tag}_mrbench_{s}"] = transfer_auc(m, XM[srcm == s], ym[srcm == s])
    res["transfer"] = t

    # 3. silhouette (cosine)
    res["silhouette"] = {"label": silhouette(X2, lab), "generator": silhouette(X2[two], isclaude[two]),
                         "context_source": silhouette(X2, src), "mrbench_label": silhouette(XM, ym)}

    # 4. domain gap
    conv = np.array([r["id"].rsplit("_", 1)[0] for r in mrb])
    Xd = np.vstack([X2, XM])
    isM = np.r_[np.zeros(len(X2), bool), np.ones(len(XM), bool)]
    gd = np.r_[grp, conv]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X2), len(XM), replace=False)  # same size as MRBench so neither pool has more neighbours
    res["domain"] = {"probe_v2_vs_mrbench": probe_cv(Xd, isM, gd),
                     "nn_to_v2": nn_cosine(XM, X2[sub]),
                     "nn_within_mrbench": nn_cosine(XM, XM, conv, conv)}

    # 5. minimal pairs
    res["pairs"] = pair_cosines(v2, X2)

    if plots:
        plot(name, X2, XM, gen, lab, ym)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{name}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def plot(name, X2, XM, gen, lab, ym, n2=1500, nm=800):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(0)
    i2, im = rng.choice(len(X2), n2, replace=False), rng.choice(len(XM), nm, replace=False)
    X = np.vstack([X2[i2], XM[im]])
    isM = np.r_[np.zeros(n2, bool), np.ones(nm, bool)]
    labels = np.r_[lab[i2], ym[im]]
    gens = np.r_[gen[i2], np.full(nm, "MRBench")]
    gcol = {"claude-haiku-5-5": "#d95f02", "gemini": "#1b9e77", "gpt-oss-120b": "#7570b3", "MRBench": "#999999"}
    for method, proj in (("pca", PCA(2, random_state=0)), ("tsne", TSNE(2, metric="cosine", init="pca", random_state=0))):
        Z = proj.fit_transform(X)
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        for g, c in gcol.items():
            k = gens == g
            ax[0].scatter(*Z[k].T, s=5, c=c, alpha=0.5, label=g)
        ax[0].set_title("by generator")
        for v, c, nm_ in ((True, "#d73027", "reveal (yes)"), (False, "#4575b4", "no reveal (no)")):
            k = (labels == v) & ~isM
            ax[1].scatter(*Z[k].T, s=5, c=c, alpha=0.5, label=nm_ + ", reveal_v2")
            k = (labels == v) & isM
            ax[1].scatter(*Z[k].T, s=12, c=c, alpha=0.8, marker="x", label=nm_ + ", MRBench")
        ax[1].set_title("by label")
        ax[2].scatter(*Z[~isM].T, s=5, c="#4575b4", alpha=0.4, label="reveal_v2")
        ax[2].scatter(*Z[isM].T, s=5, c="#d73027", alpha=0.5, label="MRBench")
        ax[2].set_title("by dataset")
        for a in ax:
            a.set_xticks([]), a.set_yticks([]), a.legend(markerscale=3, fontsize=7)
        fig.suptitle(f"{name} {method.upper()} of reply embeddings ({n2} reveal_v2 + {nm} MRBench rows)")
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{name}_{method}.png", dpi=110)
        plt.close(fig)


def table():
    rs = [json.load(open(p)) for p in sorted(OUT.glob("*.json"))]
    cols = [("probe_cv", "label"), ("probe_cv", "generator"), ("probe_cv", "generator_in_yes"), ("probe_cv", "generator_in_no"),
            ("probe_cv", "context_source"), ("transfer", "v2_val"), ("transfer", "v2_mrbench"), ("transfer", "v2_mrbench_MathDial"),
            ("transfer", "v2_mrbench_Bridge"), ("transfer", "v1_mrbench"), ("transfer", "v1_mrbench_MathDial"),
            ("transfer", "v1_mrbench_Bridge"), ("silhouette", "label"), ("silhouette", "generator"), ("silhouette", "context_source"),
            ("domain", "probe_v2_vs_mrbench"), ("domain", "nn_to_v2"), ("domain", "nn_within_mrbench"),
            ("pairs", "pair"), ("pairs", "free"), ("pairs", "pair_random"), ("pairs", "free_random")]
    print("| metric | " + " | ".join(r["embedder"] for r in rs) + " |\n|---|" + "---|" * len(rs))
    for a, b in cols:
        print(f"| {a}.{b} | " + " | ".join(f"{r[a][b]:.3f}" for r in rs) + " |")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        rng = np.random.default_rng(0)
        y = np.arange(400) % 2 == 0
        g = np.arange(400) // 4
        sep = rng.normal(size=(400, 8)) + 3 * y[:, None]
        assert probe_cv(sep, y, g) > 0.99
        assert abs(probe_cv(rng.normal(size=(400, 8)), y, g) - 0.5) < 0.12
        u = np.eye(3, dtype=np.float32)
        assert nn_cosine(u, u, [0, 1, 2], [0, 1, 2]) == 0.0 and nn_cosine(u, u) == 1.0
        assert silhouette(sep / np.linalg.norm(sep, axis=1, keepdims=True), y) > 0.05
        rows = [{"id": f"{c}_{k}", "label": l, "pair_id": p} for c, p in (("a", "a_pair"), ("b", None)) for k, l in (("guide", "no"), ("reveal", "yes"))]
        X = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float32)
        pc = pair_cosines(rows, X)
        assert pc["pair"] == 1.0 and pc["free"] == 0.0 and pc["pair_n"] == pc["free_n"] == 1
        assert key("a") != key("b") and len(key("a")) == 40
        print("ok")
        sys.exit()
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", help="embedder names from config.EMBEDDERS (default: all)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--table", action="store_true")
    a = p.parse_args()
    if a.table:
        table()
    else:
        for m in a.model or config.EMBEDDERS:
            audit(m, plots=not a.no_plots)
