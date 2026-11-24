"""Helpers for diagnosis.ipynb (MNMath, digits 0 to 2): pool loading, the two hypothesis families, the remapping of a
reading, the per-cell counts and Lemma 1 of Proposition 3, the thinning of a cell. Numpy only: a pool is the output of
a perception on a labelled set."""
import itertools, json, math, os
import numpy as np

CLASSES = (0, 1, 2)                                     # the digit classes, G
CELLS = [(a, b) for a in CLASSES for b in CLASSES]      # the 9 cells a rule reads, R = G^2
IDENTITY = CLASSES
RULES = {"control": dict(fn=lambda a, b: 2 * a + b, pos=(0, 1), Y=7, col=0),     # the control rule, never edited
         "edited": dict(fn=lambda a, b: a + b, pos=(2, 3), Y=6, col=1)}          # the rule the edits touch
T0 = {name: r["fn"] for name, r in RULES.items()}
LOG_M = math.log(len(CLASSES) ** len(CLASSES))          # log|M| = log 27: prior of a remapping, and the margin m

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POOLS = {"clean": "clean", "shortcut 1<->2": "shortcut", "dense a*b": "dense", "local (a+b) mod 3": "local",
         "exception a=2": "exception",
         **{f"{w} digit{'s' * (w == 'two')} {r}%": f"{w}_{r}" for w in ("one", "two") for r in (10, 20, 50, 70, 100)},
         "collapse 2->1": "collapse"}


def load_pool(name):
    """p [N, 4, 3] the softmax of the network on each of the 4 digits, c [N, 4] the true digits, y [N, 2] the labels of
    the two rules, meta the description of the pool (cause, corruption rate, true edit or shortcut)."""
    d = np.load(os.path.join(DATA, name + ".npz"))
    return d["p"].astype(float), d["c"].astype(int), d["y"].astype(int), json.loads(str(d["meta"]))


# --- the two families of hypotheses ---------------------------------------------------------------------------------
def remappings():
    """M: every map G -> G, the identity first; |M| = 27."""
    return [IDENTITY] + [a for a in itertools.product(CLASSES, repeat=len(CLASSES)) if a != IDENTITY]


def alpha_name(alpha):
    return "".join(map(str, CLASSES)) + "->" + "".join(map(str, alpha))


ALT_LAWS = {"a*b": lambda a, b: a * b, "(a+b)%3": lambda a, b: (a + b) % 3, "a+b+1": lambda a, b: a + b + 1,
            "max(a,b)": lambda a, b: max(a, b), "min(a,b)": lambda a, b: min(a, b), "|a-b|": lambda a, b: abs(a - b),
            "a": lambda a, b: a, "b": lambda a, b: b}
NOMINAL = {"replacement": len(ALT_LAWS), "guard row": len(CLASSES) * len(ALT_LAWS), "guard column": len(CLASSES) * len(ALT_LAWS),
           "patch": len(CELLS) * 4}                      # nominal size of each family: 8, 24, 24, 36 (4 = the other labels of a cell)


def table(fn):
    return tuple(fn(a, b) for a, b in CELLS)


def edits(rule=T0["edited"]):
    """U: the candidate edits of a rule, name -> (family, function), without those whose table equals the rule's;
    84 for a + b."""
    U = {}
    for name, law in ALT_LAWS.items():
        U[f"replace by {name}"] = ("replacement", law)
    for k in CLASSES:
        for name, law in ALT_LAWS.items():
            U[f"if a={k}: {name}"] = ("guard row", (lambda law, k: lambda a, b: law(a, b) if a == k else rule(a, b))(law, k))
            U[f"if b={k}: {name}"] = ("guard column", (lambda law, k: lambda a, b: law(a, b) if b == k else rule(a, b))(law, k))
    ymax = max(rule(a, b) for a, b in CELLS)
    for (a0, b0) in CELLS:
        for yv in range(ymax + 1):
            if yv != rule(a0, b0):
                U[f"patch ({a0},{b0})->{yv}"] = ("patch", (lambda a0, b0, yv: lambda a, b: yv if (a, b) == (a0, b0) else rule(a, b))(a0, b0, yv))
    t0 = table(rule)
    return {n: fl for n, fl in U.items() if table(fl[1]) != t0}


def edit_priors(U):
    """L(T) = -log P(T | H1) with P(T) proportional to 1 / nominal size of its family, normalised over U."""
    Z = sum(1.0 / NOMINAL[fam] for fam, _ in U.values())
    return {n: math.log(NOMINAL[fam]) + math.log(Z) for n, (fam, _) in U.items()}, Z


def diff_cells(U, rule=T0["edited"]):
    """[|U|, 9]: the cells on which each edit differs from the rule, in the order of CELLS."""
    return np.array([[fn(*cell) != rule(*cell) for cell in CELLS] for _, fn in U.values()])


# --- readings under a remapping --------------------------------------------------------------------------------------
def translate(p, alpha):
    """Hypothesis alpha: the network reads alpha[g] when the true concept is g. p [N, 3] a distribution over readings;
    returns q [N, 3], the same mass brought back to the true concepts, shared equally among the concepts that produce
    a reading, and m [N], the orphan mass of the readings no concept produces. q.sum(1) + m == 1."""
    alpha = np.asarray(alpha); counts = np.bincount(alpha, minlength=p.shape[1])
    q = p[:, alpha] / counts[alpha]; m = p[:, counts == 0].sum(1)
    return q, m


def preimages(alpha, cell):
    return [c for c in CELLS if (alpha[c[0]], alpha[c[1]]) == tuple(cell)]


def read_cells(p, rule):
    """The cell the network reads for a rule (argmax per position) and the mass it puts on it."""
    i, j = RULES[rule]["pos"]
    r = np.stack([p[:, i].argmax(-1), p[:, j].argmax(-1)], 1); conf = p[:, i].max(-1) * p[:, j].max(-1)
    return r, conf


def read_index(p, rule):
    """The read cell of each observation for a rule, as its index in CELLS."""
    r, _ = read_cells(p, rule)
    return r[:, 0] * len(CLASSES) + r[:, 1]


def cell_mass(p, rule):
    """J [N, 9]: the mass of each observation on each cell of a rule (product of the two positions)."""
    i, j = RULES[rule]["pos"]
    return np.einsum("ia,ib->iab", p[:, i], p[:, j]).reshape(len(p), -1)


# --- the counts of Proposition 3, per (rule, cell) ---------------------------------------------------------------------
def counts(p, c, y, delta):
    """N_c^sane, N_c^mis [2, 9]: confident sane and confident misread observations read in each cell of each rule,
    q_c [2, 9] the largest share of a single wrong label among the misread ones, and the masks conf [2, N] of the
    confident observations. An observation is sane if read correctly with the label of T0, misread if its read cell is
    not its true cell."""
    rules = list(RULES); Ns = np.zeros((2, 9)); Nm = np.zeros((2, 9)); Nm_lab = np.zeros((2, 9, 8)); conf = np.zeros((2, len(y)), bool)
    for ri, rule in enumerate(rules):
        r, m = read_cells(p, rule); conf[ri] = m >= 1 - delta; true = c[:, list(RULES[rule]["pos"])]
        ok = (r == true).all(1); yo = y[:, RULES[rule]["col"]]
        for i in np.where(conf[ri])[0]:
            k = CELLS.index(tuple(r[i]))
            if ok[i] and T0[rule](*true[i]) == yo[i]: Ns[ri, k] += 1
            elif not ok[i]: Nm[ri, k] += 1; Nm_lab[ri, k, yo[i]] += 1
    T0lab = np.array([[T0[rule](*cell) for cell in CELLS] for rule in rules])
    wrong = Nm_lab.copy()
    for ri in range(2):
        for k in range(9): wrong[ri, k, T0lab[ri, k]] = 0
    q_c = np.where(Nm > 0, wrong.max(2) / np.maximum(Nm, 1), 0.0)
    return Ns, Nm, q_c, conf


def identity_lemma(Ns, Nm, q_c, Ch, J, P, kp, ridx, zeta, eps, gamma):
    """Lemma 1 of the paper (alpha* = id), for every alpha != id: the sane observations alpha breaks (cells R_alpha)
    against what it can repair in the cells it touches (A_alpha), their costs, and the leak toward A_alpha of the
    observations read elsewhere. Returns the smallest slack and the alpha that gives it. Arrays are per rule:
    Ns, Nm, q_c, Ch [2, 9]; J, P, ridx, kp per rule name; zeta, eps, gamma the constants."""
    A = remappings()[1:]; rules = list(RULES); nA = len(A)
    K = np.zeros((nA, 9)); inR = np.zeros((nA, 2, 9), bool)
    for ai, alpha in enumerate(A):
        for k, cell in enumerate(CELLS):
            pre = preimages(alpha, cell); K[ai, k] = len(pre)
            for ri, rule in enumerate(rules):
                inR[ai, ri, k] = T0[rule](*cell) not in {T0[rule](*cp) for cp in pre}
    touched = ~((K[:, None, :] == 1) & ~inR)                                           # A_alpha: preimage not unique, or another label
    theta = np.zeros(nA)
    for ai in range(nA):
        for ri, rule in enumerate(rules):
            Aset = touched[ai, ri]; outside = ~Aset[ridx[rule]]
            theta[ai] += np.log(1 + J[rule][outside][:, Aset].sum(1) / (P[rule][outside] + kp[rule])).sum()
    broken = (inR * Ns[None]).sum((1, 2)) * (zeta - eps)                                # (i) sane observations read in R_alpha
    untouched = ((touched & ~inR) * Ns[None]).sum((1, 2)) * eps                         # (ii) sane read in A_alpha \ R_alpha
    Kb = K[:, None, :]
    sigma = np.where((Kb == 1) & inR, q_c[None], np.where(Kb >= 2, np.minimum(1.0, Kb * q_c[None]), 0.0))
    repaired = zeta * (sigma * Nm[None]).sum((1, 2))                                    # (iii) the repaired misread ones, zeta more
    costs = (touched * (Nm[None] * gamma + Ch[None])).sum((1, 2))                       # (iv), (v) gamma per misread one, the hesitating
    slack = broken - untouched - repaired - costs - theta                               # (vi) the leak
    ia = int(slack.argmin())
    return float(slack[ia]), alpha_name(A[ia])


# --- rare cells --------------------------------------------------------------------------------------------------------
def thin(c, cell, k, seed):
    """The rare-cell protocol: the indices of a pool in which only k of the observations whose true cell on the edited
    rule is `cell` are kept, drawn with `seed`, the rest of the pool unchanged."""
    i, j = RULES["edited"]["pos"]; inside = np.where((c[:, i] == cell[0]) & (c[:, j] == cell[1]))[0]
    kept = np.random.default_rng(seed).choice(inside, size=min(k, len(inside)), replace=False)
    return np.sort(np.concatenate([np.setdiff1d(np.arange(len(c)), inside), kept])).astype(int)


def show(rows, cols, fmt=None):
    """A small text table: rows = list of dicts, cols = the keys to print."""
    fmt = fmt or {}
    w = {k: max(len(k), *(len(fmt.get(k, "{}").format(r[k])) for r in rows)) for k in cols}
    print("  ".join(k.ljust(w[k]) for k in cols))
    for r in rows: print("  ".join(fmt.get(k, "{}").format(r[k]).ljust(w[k]) for k in cols))
