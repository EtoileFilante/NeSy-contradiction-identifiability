# When is a contradiction perceptual or symbolic?

Companion notebooks of the paper *When Is a Contradiction Perceptual or Symbolic? Identifiability in Neuro-Symbolic
Models*. They run the diagnosis on the pools of the paper, check the conditions of its three propositions and apply
its support check to pools with a rare cell. Executed outputs are included; every number matches the paper.

* `diagnosis.ipynb`: MNMath. The score, the verdict, the conditions of the three propositions (wrong rule, reasoning
  shortcut, unstructured perception) and the support check, with the explanations next to the code.
* `kand_logic.ipynb`: Kand-Logic. The same verdict and conditions on the second domain, with what changes explained;
  it reproduces the Kand-Logic table of the paper.
* `helper.py`: for MNMath, loading, the enumeration of the two hypothesis families (27 concept remappings, 84 rule
  edits with their priors), the per-cell counts, Lemma 1 of Proposition 3 and the thinning of a cell for the rare-cell
  protocol.
* `kand_helper.py`: for Kand-Logic, the domain (165 cells, 729 remappings, 183 edits), loading, the score tables and
  the three conditions with multisets as cells.
* `data/`: the output of the perception on each MNMath pool, one `.npz` per pool with `p` the softmax of the network
  on each of the four digits (N × 4 × 3), `c` the true digits (N × 4), `y` the labels of the two rules (N × 2) and
  `meta` a description of the pool. Sixteen pools: clean, imposed shortcut (1 ↔ 2), collapse (2 → 1), three wrong
  rules, and one or two digits replaced on 10 to 100% of the observations.
* `data_kand/`: the same for Kand-Logic, with `p_shape` and `p_col` the softmax on the shape and the colour of each of
  the three primitives (N × 3 × 3 each), `c` the true shapes and colours (N × 3 × 2) and `y` the labels of the three
  rules (N × 3). Nineteen pools: clean, imposed shortcut (red ↔ blue), three wrong rules, and one or two primitives
  replaced on 10 to 100% of the figures.

No network and no image is needed: the pools are the readings of the trained perceptions. The training of the
perceptions, the construction of the pools and the other experiments of the paper (five and ten digits, learned
shortcuts, baselines, check on a labelled sample) are not part of this repository. Their code is available from the
authors on request.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook
```

Python 3.10 or later; each notebook runs in under a minute on a laptop.

## Licence

The code and the data are released under CC0 1.0 (see `LICENSE`).
