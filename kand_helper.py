"""Helpers for kand_logic.ipynb: the Kand-Logic domain (concepts, cells, rules, remappings, edits), pool loading, the
score tables and the three count conditions with multisets as cells (explained in diagnosis.ipynb, sections 4 to 7).
Numpy only: a pool is the output of a perception on a labelled set."""
import itertools, json, math, os
import numpy as np

# --- the domain ---------------------------------------------------------------------------------------------------------
SHAPES, COLOURS = ("sq", "ci", "tr"), ("re", "ye", "bl")                # square, circle, triangle; red, yellow, blue
CONCEPTS = [(s, c) for s in range(3) for c in range(3)]                 # concept k = 3 s + c: a shape and a colour
TRIPLES = np.array(list(itertools.product(range(9), repeat=3)))         # the 729 ordered triples of concepts, G^3
CELLS = sorted({tuple(sorted(t)) for t in TRIPLES.tolist()})            # the 165 multisets: the cells of the rules
CELL_OF = np.array([CELLS.index(tuple(sorted(t))) for t in TRIPLES.tolist()])   # ordered triple 81 i + 9 j + k -> its cell
ORDERINGS = np.bincount(CELL_OF)                                        # number of orderings of each cell

# the pattern of the benchmark, "two primitives share shape and colour and the third has another shape", as an OR of
# three AND clauses of comparisons (attribute, i, j, equal?) between the primitives i and j of a cell in sorted order
PATTERN = ("or", [("and", [("c", 0, 1, True), ("s", 0, 1, True), ("s", 0, 2, False)]),
                  ("and", [("c", 0, 2, True), ("s", 0, 2, True), ("s", 0, 1, False)]),
                  ("and", [("c", 1, 2, True), ("s", 1, 2, True), ("s", 0, 2, False)])])


def evaluate(F, cell):
    top, clauses = F; prims = [CONCEPTS[k] for k in cell]; at = {"s": 0, "c": 1}
    vals = [(all if op == "and" else any)((prims[i][at[a]] == prims[j][at[a]]) == eq for a, i, j, eq in atoms) for op, atoms in clauses]
    return int((all if top == "and" else any)(vals))


def flip(F, k=None, a=None):
    """The formula with comparison a of clause k flipped, or the connective of clause k, or the top connective."""
    top, clauses = F; clauses = [(op, list(atoms)) for op, atoms in clauses]; other = {"and": "or", "or": "and"}
    if k is None: top = other[top]
    elif a is None: clauses[k] = (other[clauses[k][0]], clauses[k][1])
    else: attr, i, j, eq = clauses[k][1][a]; clauses[k][1][a] = (attr, i, j, not eq)
    return top, clauses


def table(fn):
    return np.array([int(fn(m)) for m in CELLS])


def cell_name(m):
    return "{" + ", ".join(f"{SHAPES[CONCEPTS[k][0]]}-{COLOURS[CONCEPTS[k][1]]}" for k in m) + "}"


Y = (2, 7, 7)                                                           # labels of the pattern (edited), the colour score and the shape score (controls)
T0 = [table(lambda m: evaluate(PATTERN, m)), table(lambda m: sum(CONCEPTS[k][1] for k in m)), table(lambda m: sum(CONCEPTS[k][0] for k in m))]

# --- the two families of hypotheses -------------------------------------------------------------------------------------
MAPS = list(itertools.product(range(3), repeat=3))                      # the 27 maps of one attribute
IDENTITY = ((0, 1, 2), (0, 1, 2))
ALPHAS = [IDENTITY] + [(s, c) for s in MAPS for c in MAPS if (s, c) != IDENTITY]   # M: a map per attribute, the identity first
LOG_M = math.log(len(ALPHAS))                                           # log|M| = log 729: prior of a remapping, and the margin m


def alpha_name(alpha):
    return ", ".join(f"{attr} 012->{''.join(map(str, a))}" for attr, a in zip(("shape", "colour"), alpha) if a != (0, 1, 2)) or "id"


def remap(alpha, concepts):
    """alpha on an array of concepts: the shape through alpha[0], the colour through alpha[1]."""
    return 3 * np.asarray(alpha[0])[concepts // 3] + np.asarray(alpha[1])[concepts % 3]


OTHER_PATTERNS = {"all same colour": lambda s, c: len(c) == 1, "all colours different": lambda s, c: len(c) == 3,
                  "all same shape": lambda s, c: len(s) == 1, "all shapes different": lambda s, c: len(s) == 3,
                  "exactly two of the same shape": lambda s, c: len(s) == 2, "exactly two of the same colour": lambda s, c: len(c) == 2}
NOMINAL = {"replacement": len(OTHER_PATTERNS), "formula": 13, "patch": len(CELLS)}   # nominal size of each family


def edits():
    """U: the candidate edits of the pattern, name -> (family, label table over the cells), without those whose table
    is the pattern's; 183."""
    U = {}
    for name, pat in OTHER_PATTERNS.items():
        U[name] = ("replacement", table(lambda m: pat({CONCEPTS[k][0] for k in m}, {CONCEPTS[k][1] for k in m})))
    for k in range(3):
        for a in range(3): U[f"comparison {a} of clause {k} flipped"] = ("formula", table(lambda m: evaluate(flip(PATTERN, k, a), m)))
        U[f"connective of clause {k} flipped"] = ("formula", table(lambda m: evaluate(flip(PATTERN, k), m)))
    U["top connective flipped"] = ("formula", table(lambda m: evaluate(flip(PATTERN), m)))
    for i, m in enumerate(CELLS):
        t = T0[0].copy(); t[i] = 1 - t[i]; U["patch " + cell_name(m)] = ("patch", t)
    return {n: ft for n, ft in U.items() if (ft[1] != T0[0]).any()}


U = edits(); EDITS = list(U); E = np.array([t for _, t in U.values()]); DIFF = E != T0[0]   # E, DIFF [|U|, cells]
Z = sum(1.0 / NOMINAL[fam] for fam, _ in U.values())
PRIOR_U = np.array([math.log(NOMINAL[fam] * Z) for fam, _ in U.values()])   # L(T) = log(nominal size of the family) + log Z


def preimage_counts(T, n_labels):
    """[|M|, cells, labels]: for each remapping alpha and each read cell, the number of ordered triples alpha sends to
    an ordering of that cell, split by the label T gives them."""
    n = np.zeros((len(ALPHAS), len(CELLS), n_labels))
    for ai, alpha in enumerate(ALPHAS): np.add.at(n[ai], (CELL_OF[remap(alpha, TRIPLES) @ (81, 9, 1)], T[CELL_OF]), 1)
    return n


COUNTS = [preimage_counts(T0[r], Y[r]) for r in range(3)]
SHARES = [n / np.maximum(n.sum(2, keepdims=True), 1) for n in COUNTS]   # share of the preimages of a read cell that carry each label
K = (COUNTS[0].sum(2) / ORDERINGS).astype(int)                          # [|M|, cells] preimages of one ordering of the read cell
BROKEN = np.stack([COUNTS[r][:, np.arange(len(CELLS)), T0[r]] == 0 for r in range(3)], 1)   # [|M|, 3, cells] R_alpha: no preimage keeps the label
TOUCHED = ~((K[:, None, :] == 1) & ~BROKEN)                             # A_alpha: preimage not unique, or another label

# --- pools and scores -----------------------------------------------------------------------------------------------------
KAPPA = 1e-3; DELTAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_kand")
POOLS = {"clean": "clean", "shortcut, red <-> blue": "shortcut", "dense (all colours different)": "dense",
         "local (one comparison flipped)": "local", "exception (one cell patched)": "exception",
         **{f"{w} primitive{'s' * (w == 'two')}, {r}%": f"{w}_{r}" for w in ("one", "two") for r in (10, 20, 30, 40, 50, 70, 100)}}


def load_pool(name):
    """A pool as a dict. From the file: y [N, 3] the labels of the three rules and meta (cause, corruption rate, true
    edit or shortcut). Derived from the softmax of the network on the shape and the colour of each primitive and from
    the true concepts: J [N, cells] the mass on each cell, read [N] the cell read and conf [N] the mass on the reading,
    reading and truth [N, 3] the concept read and the true concept of each primitive."""
    d = np.load(os.path.join(DATA, POOLS[name] + ".npz")); ps, pc = d["p_shape"].astype(float), d["p_col"].astype(float)
    slot = (ps[:, :, :, None] * pc[:, :, None, :]).reshape(len(ps), 3, 9)             # each primitive over the nine concepts
    J = np.zeros((len(ps), len(CELLS))); np.add.at(J.T, CELL_OF, np.einsum("ni,nj,nk->ijkn", *slot.transpose(1, 0, 2)).reshape(729, -1))
    reading = 3 * ps.argmax(-1) + pc.argmax(-1); truth = 3 * d["c"][:, :, 0].astype(int) + d["c"][:, :, 1]
    return dict(J=J, y=d["y"].astype(int), read=CELL_OF[reading @ (81, 9, 1)], conf=(ps.max(-1) * pc.max(-1)).prod(-1),
                reading=reading, truth=truth, meta=json.loads(str(d["meta"])))


def compatible_mass(J, y, shares):
    """P [N, H]: for each figure and each hypothesis h, the mass compatible with its label, the sum over the read cells
    r of J[i, r] * shares[h, r, y_i]. Under a remapping, shares is the fraction of the preimages of r that carry the
    label; under an edit, it is 1 on the cells the edit maps to the label."""
    P = np.empty((len(y), len(shares)))
    for lab in np.unique(y): P[y == lab] = J[y == lab] @ shares[:, :, lab].T
    return P


def score_tables(pool, kappa=KAPPA):
    """s0: for each rule, the score of every figure under each (alpha, T0), [N, |M|] (column 0 is the identity);
    s1 [N, |U|]: the score of the pattern under each (id, T)."""
    J, y = pool["J"], pool["y"]; score = lambda P, n: -np.log((1 - n * kappa) * P + kappa)
    return [score(compatible_mass(J, y[:, r], SHARES[r]), Y[r]) for r in range(3)], score(compatible_mass(J, y[:, 0], np.eye(2)[E]), 2)


# --- the three count conditions (diagnosis.ipynb, sections 4 to 7), each at the delta that gives the largest slack ------
def constants(delta, kappa=KAPPA):
    eps = max(-math.log((1 - n * kappa) * (1 - delta) + kappa) for n in Y)            # the largest label set
    return -math.log(delta + kappa), eps, max(math.log(1 + delta / kappa), eps), -math.log(kappa)   # zeta, eps, gamma, zeta_bar


def follows(pool, alpha=IDENTITY):
    """The figures whose reading is alpha of their true concepts; for the identity, the figures read correctly."""
    return (np.sort(pool["reading"], 1) == np.sort(remap(alpha, pool["truth"]), 1)).all(1)


def best(slack_at):
    return max(({"delta": d, **slack_at(d)} for d in DELTAS), key=lambda r: r["slack"])


def rule_slack(pool, s0, s1, t, kappa=KAPPA):
    """Proposition 1, with T1 the edit of index t: N_alpha zeta > N eps + L(T1) + C_h + N_mis zeta_bar for every alpha."""
    y, read, ok = pool["y"], pool["read"], follows(pool)
    def at(delta):
        zeta, eps, _, zeta_bar = constants(delta, kappa); conf = pool["conf"] >= 1 - delta; sel = conf & ok
        N_alpha = sum((COUNTS[r][:, read[sel], y[sel, r]] == 0).sum(1) for r in range(3))   # per alpha: correct readings left contradicted
        C_h = (s1[~conf, t] + s0[1][~conf, 0] + s0[2][~conf, 0]).sum(); N_mis = 3 * int((conf & ~ok).sum())
        return dict(slack=N_alpha.min() * zeta - (y.size * eps + PRIOR_U[t] + C_h + N_mis * zeta_bar))
    return best(at)


def shortcut_slack(pool, s0, a, kappa=KAPPA):
    """Proposition 2, with alpha0 the remapping of index a: N_viol zeta > N eps + sum log K_i + N_mis zeta_bar + C_h + 2 log|M|."""
    y, read, fol = pool["y"], pool["read"], follows(pool, ALPHAS[a])
    def at(delta):
        zeta, eps, _, zeta_bar = constants(delta, kappa); conf = pool["conf"] >= 1 - delta; sel = conf & fol
        N_viol = sum(int((T0[r][read[sel]] != y[sel, r]).sum()) for r in (1, 2))            # on the controls: the read cell contradicts the label
        sum_log_K = 3 * np.log(np.maximum(K[a, read[conf]], 1)).sum(); N_mis = 3 * int((conf & ~fol).sum())
        return dict(slack=N_viol * zeta - (y.size * eps + sum_log_K + N_mis * zeta_bar + sum(s[~conf, a].sum() for s in s0) + 2 * LOG_M))
    return best(at)


def noise_slack(pool, s0, kappa=KAPPA):
    """Proposition 3: the per-cell condition on the pattern, and Lemma 1 for every alpha != id; the smaller slack."""
    y, read, J, ok = pool["y"], pool["read"], pool["J"], follows(pool); nC = len(CELLS); rows = np.arange(len(y))
    P = [(J * (T0[r][None, :] == y[:, r, None])).sum(1) + kappa / (1 - Y[r] * kappa) for r in range(3)]   # compatible mass under T0, plus kappa'
    leak = np.log(1 + J / P[0][:, None])                                               # [N, cells] toward each cell, on the pattern
    theta_c = leak.sum(0) - np.bincount(read, weights=leak[rows, read], minlength=nC)   # from the figures read elsewhere
    populated = np.bincount(read, minlength=nC) > 0; theta_empty = theta_c[~populated].sum()
    theta_alpha = sum(((~TOUCHED[:, r, read].T) * np.log(1 + (J @ TOUCHED[:, r].T) / P[r][:, None])).sum(0) for r in range(3))   # leak toward A_alpha
    def at(delta):
        zeta, eps, gamma, _ = constants(delta, kappa); conf = pool["conf"] >= 1 - delta; mis = conf & ~ok
        Ns = np.array([np.bincount(read[conf & ok & (T0[r][read] == y[:, r])], minlength=nC) for r in range(3)])   # [3, cells] confident sane
        Nm = np.bincount(read[mis], minlength=nC); q = np.zeros((3, nC))                 # [cells] confident misread; q_c per rule
        for r in range(3):
            lab = np.zeros((nC, Y[r])); np.add.at(lab, (read[mis], y[mis, r]), 1); lab[np.arange(nC), T0[r]] = 0
            q[r] = lab.max(1) / np.maximum(Nm, 1)
        Ch = np.array([np.bincount(read[~conf], weights=s0[r][~conf, 0], minlength=nC) for r in range(3)])        # [3, cells] hesitating, under T0
        per_cell = (Ns[0] * (zeta - eps) - q[0] * Nm * zeta - Nm * gamma - Ch[0] - theta_c - theta_empty - 2 * LOG_M)[populated].min()
        Kb = K[:, None, :]; sigma = np.where((Kb == 1) & BROKEN, q, np.where(Kb >= 2, np.minimum(1.0, Kb * q), 0.0))
        lemma = ((BROKEN * Ns).sum((1, 2)) * (zeta - eps) - ((TOUCHED & ~BROKEN) * Ns).sum((1, 2)) * eps - zeta * (sigma * Nm).sum((1, 2))
                 - (TOUCHED * (Nm * gamma + Ch)).sum((1, 2)) - theta_alpha)[1:].min()
        return dict(slack=min(per_cell, lemma), per_cell=per_cell, lemma=lemma)
    return best(at)
