"""Build the Walkthrough tab (figures, tables, verbatim code excerpts) and patch it into site/index.html.

Idempotent: an earlier Walkthrough is replaced. Code excerpts are read from src/ at build time, so re-run this
after changing the code. Numbers in the prose and figures are written here by hand from the reports, except the
real-pairs table in part 9, which is read from results/ot/pair_plot_stats_X_pca_g.csv (scripts/plot_ot_pairs.py).

    python scripts/build_walkthrough.py
"""
import csv
import html
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import walkthrough_figs as F  # noqa: E402

SITE = ROOT / "site" / "index.html"


# ------------------------------------------------------------------ helpers
def fig(svg, caption):
    return f'<figure class="block">\n<div class="diagram scroll">{svg}</div>\n<figcaption>{caption}</figcaption>\n</figure>'


def imgfig(src, alt, caption):
    """A PNG from figures/. The path is relative to the published page; scripts/wrap_preview.sh rewrites it for site/notebook.html."""
    assert (ROOT / src).exists(), f"{src} is missing: run scripts/plot_ot_pairs.py"
    return f'<figure class="block">\n<img src="{src}" alt="{alt}" loading="lazy">\n<figcaption>{caption}</figcaption>\n</figure>'


def ot_pair_rows():
    """Rows for the real-pairs table, read from the numbers scripts/plot_ot_pairs.py saved next to its figure."""
    rows = []
    with open(ROOT / "results" / "ot" / "pair_plot_stats_X_pca_g.csv") as f:
        for r in csv.DictReader(f):
            sq = float(r["shift_len"]) ** 2
            rows.append([f'{r["cond"]} ({r["tier"]})', f"{sq:.0f}" if sq >= 10 else f"{sq:.1f}", f'{float(r["ot_sq_len"]):.0f}',
                         f'{float(r["random_sq_len"]):.0f}', f'{100 * float(r["ot_visible_share"]):.0f}%'])
    return rows


def code(rel, start, end, what, open_=False):
    """Verbatim source between the first line containing `start` and the next line containing `end` (inclusive)."""
    lines = (ROOT / rel).read_text().splitlines()
    a = next(i for i, l in enumerate(lines) if start in l)
    b = next(i for i in range(a, len(lines)) if end in lines[i])
    body = html.escape(textwrap.dedent("\n".join(lines[a:b + 1])))
    return (f'<details class="code"{" open" if open_ else ""}><summary><span class="mono">{rel}</span> · {what} · lines {a + 1}–{b + 1}</summary>'
            f'<pre>{body}</pre></details>')


def mono(text):
    return f'<span class="mono">{text}</span>'


def nlines(*rels):
    """Total lines of the given files, counted when the page is built."""
    return f"{sum(len((ROOT / r).read_text().splitlines()) for r in rels):,}"


SCRIPTS = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "scripts").iterdir() if p.suffix in (".py", ".sh"))
TESTS = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("*.py"))


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap {cls}"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def trade(rows):
    return table(["Decision", "Chosen", "Turned down", "Why", "What it costs"], rows, "trade")


FIT, APP, PICK, SCORE, NO = ('<span class="lk fit">fitted here</span>', '<span class="lk app">passed through</span>',
                             '<span class="lk pick">used to choose</span>', '<span class="lk score">scored here</span>', '<span class="lk no">never</span>')

SECTIONS = [("wk-repo", "Where the 215k lines are"), ("wk-idea", "The idea"), ("wk-data", "The data"), ("wk-split", "The split and leakage"),
            ("wk-tensors", "From counts to a cell state"), ("wk-emb", "Week 1: describing a gene"), ("wk-decoder", "Week 1: the decoder"),
            ("wk-eval", "Week 1: scoring and results"), ("wk-ot", "Week 2: pairing cells"), ("wk-actions", "Week 2: action vectors"),
            ("wk-tests", "Week 2: tests and the length problem"), ("wk-status", "Where it stands")]

# ------------------------------------------------------------------ content
parts = []
P = parts.append

P('''<section class="week" id="walk" aria-label="Walkthrough">
<div class="wrap walk">
  <div class="hero">
    <div class="eyebrow">Walkthrough · everything so far, in order</div>
    <p class="q">From a raw count matrix to action vectors: each step, its tensors, the choices behind it and what it showed.</p>
    <p class="lede">Read top to bottom. Each part says what was done, shows the shape of the data going in and out, lists the options that were turned down, and opens the code that does it. The Week 1 and Week 2 tabs hold the full result tables.</p>
  </div>
  <ol class="walk-toc">''' + "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in SECTIONS) + "</ol>")

# ---- 1 repo
P(f'''
  <div class="block" id="wk-repo">
    <div class="kicker">1 · The repository</div>
    <h2>Where the 215k lines are</h2>
    <p class="lede">The commit added 214,959 lines, and 2,931 of them are Python. Git counts every line of every text file, and this project stores its cell lists and results as text.</p>
    {fig(F.fig_lines(), "<b>Committed lines by kind.</b> Top: all 214,959. Bottom: the last 5% enlarged. The dark block is all of the code.")}
    <ul class="clean">
      <li><b><span class="mono">splits.json</span>, 91,261 lines.</b> It writes each of the 90,867 cell barcodes on its own line, so the split can be read and diffed.</li>
      <li><b><span class="mono">results/</span>, 79,326 lines.</b> Metric tables with one row per perturbation, method, seed and metric, plus the ranked gene lists used for scoring.</li>
      <li><b><span class="mono">embeddings/gene_universe.csv</span>, 33,695 lines.</b> One row per measured gene.</li>
      <li><b>Python, 2,931 lines</b> in 9 modules and 5 scripts. Without comments and docstrings it is about 2,200. The model itself is 30 lines; most of the rest is label parsing, gene-name matching, scoring and run bookkeeping.</li>
    </ul>
    {fig(F.fig_map(), "<b>How the code is laid out.</b> Three shared modules prepare the data. Week 1 and Week 2 each add three. The Week 2 actions go back through the Week 1 decoder for one test.")}
    {table(["Module", "Lines", "What it does", "Start reading at"], [
        [mono("data.py"), nlines("src/pertact/data.py"), "Reads the scPerturb file, turns guide labels into control / single / double, filters cells, normalises, picks genes on training cells", '<span class="mono">build_qc</span>, <span class="mono">build_processed</span>'],
        [mono("splits.py"), nlines("src/pertact/splits.py"), "Puts every cell in train, validation or test, and asserts that no double lands in train or validation", '<span class="mono">make_splits</span>'],
        [mono("encoders.py"), nlines("src/pertact/encoders.py"), "Fits PCA-50 and scVI on training cells, then encodes every cell", '<span class="mono">fit_pca</span>, <span class="mono">fit_scvi</span>'],
        [mono("embeddings.py"), nlines("src/pertact/embeddings.py"), "Builds the four Week 1 gene-embedding tables, and GenePT / STRING vectors for every measured gene", '<span class="mono">build_pca_loadings</span>'],
        [mono("decoders.py"), nlines("src/pertact/decoders.py"), "The model (latent additive and decoder-only), its training loop and prediction", '<span class="mono">LatentAdditive</span>, <span class="mono">train_decoder</span>'],
        [mono("metrics.py"), nlines("src/pertact/metrics.py"), "Scoring genes, the metrics, the fixed references of a split and the noise ceiling", '<span class="mono">EvalRef</span>'],
        [mono("runs.py"), nlines("src/pertact/runs.py"), "Baselines, decoder tuning, the Week 1 runs, sequential composition and rescoring", '<span class="mono">Runner</span>'],
        [mono("evaluate.py"), nlines("src/pertact/evaluate.py"), "The command line for the scoring and the runs", '<span class="mono">main</span>'],
        [mono("ot.py"), nlines("src/pertact/ot.py"), "Pairs controls with perturbed cells by entropic OT (Sinkhorn-Knopp from the POT library), with its own checks", '<span class="mono">pair</span>, <span class="mono">run_space</span>'],
        [mono("actions.py"), nlines("src/pertact/actions.py"), "Sets of action vectors: storage, the mean shift and the two rescalings", '<span class="mono">mean_shift_actions</span>'],
        [mono("infonce.py"), nlines("src/pertact/infonce.py"), "Trains the action vectors with the contrastive loss; temperature and anchor grids", '<span class="mono">infonce_loss</span>, <span class="mono">train_actions</span>'],
        [mono("retrieval.py"), nlines("src/pertact/retrieval.py"), "The forward and reverse tests of any set of action vectors", '<span class="mono">instance_retrieval</span>, <span class="mono">identification</span>'],
        [mono("decoder_test.py"), nlines("src/pertact/decoder_test.py"), "The decoder test: frozen action vectors through the Week 1 decoders, on validation singles or also on the test doubles", '<span class="mono">action_smoke</span>'],
        [mono("doubles.py"), nlines("src/pertact/doubles.py"), "Week 3: the sum of two action vectors scored on the test doubles, in the state space", '<span class="mono">score_conditions</span>'],
        [mono("scripts/") + f" ({len(SCRIPTS)} files)", nlines(*SCRIPTS), "Report tables, the perturbation map, Week 2 diagnostics and summaries, the OT pair plot, the output check, and the builders of this site", ""],
        [mono("tests/") + f" ({len(TESTS)} files)", nlines(*TESTS), "Fast tests on small synthetic inputs", ""]])}
  </div>''')

# ---- 2 idea
P(f'''
  <div class="block" id="wk-idea">
    <div class="kicker">2 · The idea</div>
    <h2>A perturbation as an arrow that moves a cell</h2>
    <p class="lede">Each cell gets a <b>state</b>: its position in a compressed version of its expression profile, 50 or 30 numbers. Switching a gene on is treated as an <b>action</b>: one vector f(a), the same for every cell, added to that state.</p>
    {fig(F.fig_idea(), "<b>The model and the test it is built for.</b> Left: f(a) carries control cells to where cells with gene a end up. Right: if actions are real vectors they can be added, so f(a) + f(b) predicts a double perturbation without measuring it. The gap between the ring and the measured cells is the interaction: what addition misses.")}
    <ul class="clean">
      <li><b>Why it matters.</b> A lab can measure a few hundred single perturbations but not all of their pairs. Vectors that add would predict the pairs for free.</li>
      <li><b>Week 1</b> built the test bed with four borrowed ways of describing a gene and one decoder, and scored them on held-out doubles.</li>
      <li><b>Week 2</b> learned the action vectors themselves from single perturbations, and tested them on held-out cells of those singles. It has not yet tested doubles.</li>
    </ul>
  </div>''')

# ---- 3 data
P(f'''
  <div class="block" id="wk-data">
    <div class="kicker">3 · The data</div>
    <h2>Why this dataset, and which cells were kept</h2>
    <p class="lede">Norman et al. 2019: K562 cells in which CRISPRa switches on one gene, or two at once. Each cell's expression is read by single-cell RNA sequencing, along with the guide it carries.</p>
    <ul class="clean">
      <li><b>It measures doubles and their singles.</b> 127 gene pairs are measured, and both genes of every pair are also measured alone. The composition test needs the answer to exist: predict a + b from a and b, then compare with real a + b cells.</li>
      <li><b>It is the usual benchmark for this question,</b> so the numbers can be set against published baselines.</li>
      <li><b>Raw counts are available,</b> so every preprocessing choice could be made on training cells only.</li>
      <li><b>What it cannot tell us.</b> It is one cell line and 105 genes. Its doubles are mostly close to additive, which makes "just add the two singles" hard to beat. No gene is absent from training, so nothing here tests unseen genes.</li>
    </ul>
    {fig(F.fig_funnel(), "<b>From the downloaded file to the cells used.</b> The parsed labels agree with scPerturb's own perturbation column for all 111,445 cells. The four dropped doubles had 46 to 49 cells each.")}
    {trade([
        ["Dataset", "Norman 2019, K562 CRISPRa", "Larger single-only screens", "Only doubles with both singles measured can test composition", "One cell line, 105 genes, near-additive doubles"],
        ["File", "scPerturb's raw integer counts", "The preprocessed GEARS file", "Normalisation, gene choice and scVI could all be done on training cells; a ready-made matrix arrives already normalised and gene-filtered", "Guide labels had to be parsed here (checked against scPerturb: 111,445 of 111,445 agree)"],
        ["Cells with several guide identities", "Dropped (13,339)", "Keep them", "Their label is ambiguous, and their median count is 20.1k against 13.4k, the signature of two cells in one droplet", "13% fewer cells"],
        ["Small perturbations", "Need at least 50 cells", "Keep all", "A mean profile from under 50 cells is too noisy to score against", "4 of 131 doubles lost"]])}
  </div>''')

# ---- 4 split
P(f'''
  <div class="block" id="wk-split">
    <div class="kicker">4 · The split and leakage</div>
    <h2>Which cells each step was allowed to see</h2>
    <p class="lede">Every cell is in exactly one of three sets. Doubles are never in training or validation. <span class="mono">splits.py</span> asserts both facts, and re-running it gives the same file byte for byte.</p>
    {fig(F.fig_split(), "<b>The split, by kind of cell.</b> Each perturbation's cells are split 80 / 20 on their own, so every single is in both train and validation. Controls are split inside each gem group (sequencing batch), so every set has controls from every batch.")}
    <h3>What touched what</h3>
    {table(["Step", "Train", "Validation", "Test"], [
        ["Gene filter and the 5,000 variable genes", FIT, NO, NO],
        ["PCA-50 and scVI state encoders", FIT, APP, APP],
        ["PCA-loadings gene embedding", FIT, NO, NO],
        ["Scoring genes for singles (top-20 changed)", FIT, NO, NO],
        ["Decoder weights", FIT, NO, NO],
        ["Decoder early stopping and the 9-setting grid", NO, PICK, NO],
        ["OT pairs", "paired among train cells", "paired among validation cells", NO],
        ["Mean-shift and InfoNCE actions, the length scales, the anchor weight", FIT, NO, NO],
        ["InfoNCE early stopping, temperature, which rescaled variant to carry on", NO, PICK, NO],
        ["Week 2 scores: retrieval, identification, decoder test", NO, SCORE, NO],
        ["Week 1 final scores on doubles", NO, NO, SCORE]])}
    <p class="tcap"><b>How to read it.</b> <i>Fitted here</i>: numbers were estimated from these cells. <i>Passed through</i>: the cells were encoded by a model fitted elsewhere. <i>Used to choose</i>: a setting was picked by its score on these cells. The scoring genes of each double come from its own test cells against test controls; that defines the score and trains nothing.</p>
    <ul class="clean">
      <li><b>No leakage into fitting.</b> No validation or test cell was used to estimate anything.</li>
      <li><b>Validation is not a test set.</b> Settings were chosen on it, so its scores run slightly high. They are fine for comparing variants and should not be quoted as final results.</li>
      <li><b>The test set chose nothing.</b> Week 1 scored its final models on it once. Week 2 has not opened it.</li>
    </ul>
    {table(["Case", "Cells", "Perturbation", "Tested here?"], [
        ["Validation", "new", "a single seen in training", "Yes: every Week 2 score"],
        ["Test", "new", "a new pair a + b, with a and b each seen alone", "Yes in Week 1; not yet for Week 2 actions"],
        ["New gene", "new", "a gene never perturbed in training", "No: one-hot and per-gene actions have no vector for it"]])}
    {trade([
        ["Where doubles go", "All 127 to test", "Some doubles in training, as some benchmarks do", "The question is whether singles alone predict doubles", "No model can learn an interaction, and validation cannot contain a double"],
        ["Validation set", "20% of the cells of every single", "Hold out whole perturbations", "Every test double needs both genes in training; holding out a single would also remove its doubles", "Validation only tests new cells of known perturbations, an easier task than the test"],
        ["Controls", "70 / 15 / 15 inside each gem group", "One shared control pool", "Predictions for validation and test start from controls no model was fitted on", "Fewer training controls (6,781)"]])}
    {code("src/pertact/splits.py", "def make_splits", "double cells leaked into train/val", "make_splits")}
  </div>''')

# ---- 5 tensors
P(f'''
  <div class="block" id="wk-tensors">
    <div class="kicker">5 · Tensors</div>
    <h2>From counts to a cell state</h2>
    <p class="lede">The count matrix shrinks in three steps to the 5,028 genes every model sees. Two encoders then compress each cell to a short vector, the state that actions act on. Each fitted step uses training cells only.</p>
    {fig(F.fig_tensors(), "<b>Shapes at every step.</b> Rectangles narrow as genes are dropped (not to scale). The 28 added genes are perturbation targets that were not among the 5,000 most variable; CRISPRa raises the target itself, so it has to be in the matrix.")}
    {trade([
        ["Normalisation", "Counts per 10,000, then log1p, using all 33,694 genes", "Normalise after picking genes", "A cell's depth should not depend on which genes were kept", "None"],
        ["Gene set", "5,000 most variable (seurat_v3, on training counts, batch-aware) + 28 targets", "All 17,285 expressed genes", "Most genes barely vary; they add noise and a 3× larger output layer", "Effects on dropped genes are invisible"],
        ["State encoder is frozen", "Fit once, never trained again", "Train the encoder with the actions", "An action is only a fixed vector if the space is fixed; otherwise encoder and actions can adapt to each other", "The space is not tuned for perturbations"],
        ["Which state space", "Both PCA-50 and scVI-30", "One of them", "PCA is linear and deterministic but keeps 19.4% of the variance. scVI is nonlinear, models counts and corrects for batch, but its training is random. A result that holds in both does not depend on the choice", "Every Week 2 run is done twice"]])}
    {code("src/pertact/data.py", "# rarely-detected genes, on training cells", "keep = cand.var_names", "gene selection on training cells")}
    {code("src/pertact/encoders.py", "def fit_pca", 'adata.obsm["X_pca_g"]', "fit_pca: fit on train, project everything")}
  </div>''')

# ---- 6 embeddings
P(f'''
  <div class="block" id="wk-emb">
    <div class="kicker">6 · Week 1: describing a gene</div>
    <h2>Four borrowed gene vectors</h2>
    <p class="lede">Week 1 learned no actions. It took four existing ways of turning a gene into a vector and asked which composes best when fed to the same decoder.</p>
    {table(["Embedding", "Table shape", "Where the numbers come from", "Covers an unseen gene?", "Weak point"], [
        ["One-hot", '<span class="mono">[105, 105]</span>', "Nothing: gene i is the i-th unit vector", "No", "Knows nothing about the gene; the free reference"],
        ["PCA loadings", '<span class="mono">[105, 32]</span>', "This dataset's training cells: average each training single, take the top 32 components, read off the target gene's loadings", "Yes, any of the 5,028 model genes", "105 genes share 32 directions"],
        ["GenePT", '<span class="mono">[105, 1536]</span>', "Text: a language-model embedding of the gene's NCBI summary", "Yes, 22,640 of 33,694 measured genes", "Knows the literature, not this cell line"],
        ["STRING node2vec", '<span class="mono">[93, 64]</span>', "The protein-interaction graph (edges scoring ≥ 700), embedded by random walks", "Yes, 16,165 genes", "12 of the 105 targets have no high-confidence edge and get no vector"]])}
    <ul class="clean">
      <li><b>Missing vectors are not imputed.</b> A condition that touches an uncovered gene is left out of that method's training, and results are also reported on the covered subset.</li>
      <li><b>One rescale per table.</b> Each table is multiplied by a single constant so its average row length is 1. That keeps the four on a common scale without changing any table's geometry.</li>
    </ul>
  </div>''')

# ---- 7 decoder
P(f'''
  <div class="block" id="wk-decoder">
    <div class="kicker">7 · Week 1: the decoder</div>
    <h2>The model, with its tensors</h2>
    <p class="lede">One small network turns a control cell's state plus a gene vector into a full expression profile. It is the test bed: the gene vectors change between runs and everything else stays fixed.</p>
    {fig(F.fig_decoder(), "<b>Latent additive decoder, shapes for one batch.</b> d is the width of the gene vector: 105 for one-hot, 32 for PCA loadings, 50 or 30 for Week 2 actions. With one-hot the network has 7.59 M weights, 6.34 M of them in the output MLP. Because W has no bias, a double's shift is exactly the sum of its two singles' shifts.")}
    <h3>How one training batch is built</h3>
    <ol class="steps">
      <li><div><h3>Pick 256 target cells</h3><p>From the 44,551 training cells: 37,770 perturbed with one gene and 6,781 controls.</p></div><span class="shape">x: [256, 5028]</span></li>
      <li><div><h3>Give each a starting control</h3><p>A random training control from the same gem group. The two cells are not a matched pair: nothing in Week 1 says which control a perturbed cell came from.</p></div><span class="shape">s_ctrl: [256, 50]</span></li>
      <li><div><h3>Look up the gene vector</h3><p>The target's row of the embedding table. A control target gets zeros, so "no perturbation" is exactly e = 0.</p></div><span class="shape">e: [256, d]</span></li>
      <li><div><h3>Predict and compare</h3><p>Forward pass, mean squared error against the target cell's log-normalised expression, one AdamW step.</p></div><span class="shape">x̂: [256, 5028] → loss: scalar</span></li>
      <li><div><h3>Stop on validation loss</h3><p>After each epoch the same loss is measured on validation cells. Training stops after 8 epochs without improvement and keeps the best weights.</p></div><span class="shape">at most 150 epochs</span></li>
    </ol>
    <p><b>Prediction.</b> For each condition, 500 control states are drawn from that split's own control pool and decoded; their average is the predicted profile, <span class="mono">[5028]</span>. A double uses e(a) + e(b).</p>
    {trade([
        ["Architecture", "Latent additive: dec(enc(s) + W·e)", "A model with interaction terms", "It is the simplest model in which composition is explicit, and everything after the gene vector is identical for every embedding", "It can only produce an interaction through the decoder's nonlinearity, without ever seeing a double"],
        ["W", "Linear, no bias", "An MLP on the gene vector", "e = 0 then means exactly no perturbation, and two shifts add exactly", "Less expressive per gene"],
        ["Starting control", "Random, same gem group", "A matched control", "No pairing exists in the data, and Week 1 was a baseline", "The state says nothing about the target cell, so the decoder learns to ignore it (section 8)"],
        ["Loss", "Mean squared error on log-normalised expression", "A count likelihood", "It matches how the predictions are scored", "On genes that are mostly zero it rewards predicting the average"],
        ["Settings", "9-setting grid on validation singles, with one-hot, then fixed: hidden 1024, latent 128, no dropout, learning rate 3e-4", "Tune per embedding", "One shared setting keeps the comparison fair and uses validation once", "Tuned with one-hot, so it may slightly favour it. Without dropout the grid's scores span 0.951 to 0.965"]])}
    {code("src/pertact/decoders.py", "def mlp(sizes, dropout)", "return self.dec(self.latent(e, s))", "the whole model: mlp and LatentAdditive", True)}
    {code("src/pertact/decoders.py", "for ep in range(hp[", "run += loss.item()", "one training epoch")}
    {code("src/pertact/decoders.py", "def condition_matrix", "return out", "a double's vector is the sum of its genes' rows")}
  </div>''')

# ---- 8 eval
P(f'''
  <div class="block" id="wk-eval">
    <div class="kicker">8 · Week 1: scoring and results</div>
    <h2>How a prediction is scored, and what Week 1 found</h2>
    <p class="lede">Every number in the result tables is computed per condition from average profiles, then averaged over conditions and over 3 seeds.</p>
    {fig(F.fig_metric(), "<b>The headline score for one condition.</b> The 20 genes are the condition's most changed against controls (Wilcoxon test). For singles they are chosen on training cells. A second version drops the activated genes themselves, which any model that knows the target gets for free.")}
    <ul class="clean">
      <li><b>Why the 20 most changed genes.</b> Over all 5,028 genes the score is dominated by genes that do not move.</li>
      <li><b>Its blind spot.</b> Correlation ignores size: a prediction half as large scores the same. Mean squared error is reported alongside.</li>
      <li><b>References with no model.</b> "Add the two singles" is control + Δa + Δb, with each Δ measured on training cells. A split-half noise ceiling (0.997 here) is the best any predictor could reach against noisy measured means.</li>
    </ul>
    {fig(F.fig_w1(), "<b>Test result: 127 held-out doubles.</b> Correlation on the 20 most changed genes, 3-seed means. No embedding reaches the dashed line. The noise ceiling, 0.997, is off the right edge.")}
    <ul class="clean">
      <li><b>Nothing beats adding the two singles.</b> The best models, one-hot at 0.960 and 0.963, sit just under 0.968.</li>
      <li><b>Nothing captures the interaction.</b> On the part addition cannot explain, models correlate 0.30 to 0.39 with the truth. "Predict no change" gets 0.375, and the ceiling is 0.89.</li>
      <li><b>One-hot composes best, PCA loadings worst.</b> PCA loadings are fine on singles (0.966) and fall to 0.923 on doubles, because 105 genes share 32 directions and their sums land where the decoder never trained.</li>
      <li><b>The decoder ignores its starting state.</b> Latent additive and decoder-only differ by at most 0.008. Applying two perturbations one after the other returns only the second.</li>
      <li><b>Validation singles cannot separate embeddings.</b> On the genes it covers, each scores 0.95 to 0.97, the same as the plain training-cell mean (0.968).</li>
    </ul>
    <p><b>What this set up for Week 2.</b> The state was useless because controls were random, so Week 2 pairs each perturbed cell with the controls it plausibly came from. The gene vectors were borrowed, so Week 2 learns them directly as actions in the state space.</p>
    {code("src/pertact/metrics.py", "def metrics(self, c, pred)", "return out", "EvalRef.metrics: every score for one condition")}
  </div>''')

# ---- 9 OT
P(f'''
  <div class="block" id="wk-ot">
    <div class="kicker">9 · Week 2: pairing cells</div>
    <h2>Which controls did each perturbed cell come from?</h2>
    <p class="lede">Sequencing destroys the cell, so no perturbed cell has a measured "before". Optimal transport supplies a guess: move the cloud of controls onto the cloud of perturbed cells with the least total movement, and read off who went where. The entropic version blurs the answer, so each perturbed cell gets a few likely controls with weights.</p>
    {fig(F.fig_ot(), "<b>One pairing problem with its tensors.</b> In the plan, darker means more of that control's mass goes to that perturbed cell. Training and validation cells are paired separately, each with their own controls, in both state spaces: 3,348 problems in all.")}
    <ul class="clean">
      <li><b>The solver is the POT library's.</b> The pairing calls POT's log-domain Sinkhorn-Knopp. It is checked two ways: on a synthetic cloud shifted by a known vector, the true source is the top partner for 100% of cells, and shifting every control leaves the plan unchanged, as it must. Until 11 November the pairs came from a hand-written solver, which agreed with POT to 1e-8 on random problems. Its pairs and results are kept in <span class="mono">results/legacy_solver/</span>. On the real cells the two solvers give a different set of partners for at most 0.16% of the cells in a file.</li>
      <li><b>On the real data</b> all problems converged. OT partners are about half as far away as random controls of the same gem group: 56.6 against 109.4 in PCA-50, 14.5 against 30.6 in scVI-30.</li>
      <li><b>The pairs are blurry.</b> In training, each perturbed cell spreads over 60 to 75 controls before the top 5 are kept.</li>
      <li><b>The pairs are an assumption.</b> "Cells move as little as possible" cannot be checked against a measurement, because none exists.</li>
    </ul>
    <h3>The pairs on real cells</h3>
    <p>The figure above is a sketch. This one is drawn from the saved pairs: the training cells of one batch (gem group 1) for a strong, a medium and a weak perturbation. The top row joins each perturbed cell to its 5 OT partners. The bottom row joins the same cells to 5 random controls, which is how Week 1 paired them.</p>
    {imgfig("figures/ot_pairs_X_pca_g.png", "Six scatter panels of control and perturbed cells for CEBPA, AHR and BCL2L11, with lines joining each perturbed cell to five controls: OT partners in the top row, random controls in the bottom row", "<b>OT pairs against random pairs, PCA-50 space.</b> Each panel is a 2-D slice of the 50 dimensions. Across is the direction of the mean shift, with the control mean at 0. Up is the direction in which the cells spread most after that. All six panels share one scale. Pairs are computed in all 50 dimensions, so each panel says how much of the pair length the slice shows.")}
    <ul class="clean">
      <li><b>CEBPA, strong.</b> OT lines run roughly parallel to the mean shift: a cell high in the cloud is paired with controls that are also high. Random lines cross each other.</li>
      <li><b>CEBPA cells that did not respond.</b> A handful of CEBPA cells sit inside the control cloud. OT pairs them with nearby controls, so their pairs say "almost no change".</li>
      <li><b>AHR and BCL2L11, medium and weak.</b> The mean shift is shorter than the cloud is wide, so the pairs are mostly local matches.</li>
    </ul>
    {table(["Perturbation", "Mean shift, squared", "Mean squared pair length, OT", "Same, random controls", "Share of the OT length visible in the slice"], ot_pair_rows())}
    <p>For CEBPA the mean shift is most of every pair's length, so OT can shorten the pairs only a little. For the other two the mean shift is a small part. The rest is the difference between one cell and its partner that the mean shift does not explain, and most of it lies outside the slice, in the other 48 dimensions. OT halves it, but it stays several times larger than the shift.</p>
    <ul class="clean">
      <li><b>What the picture leaves out.</b> It shows one batch and three perturbations out of 840 training problems. Only the 5 heaviest partners per cell are drawn, while the full plan spreads each cell over 60 to 75 controls. The slice is chosen separately for each perturbation, so the axes point in different directions in each column. The random pairs are one draw.</li>
      <li><b>To redraw it:</b> <code>python -I scripts/plot_ot_pairs.py</code>, with <code>--space X_scvi30</code>, <code>--conds</code> or <code>--gem</code> for other views.</li>
    </ul>
    {trade([
        ["How to pair", "Optimal transport", "Random control (Week 1); nearest control", "Nearest-neighbour lets many cells pick the same few controls and ignores the overall shift. OT uses every control equally and is unchanged if the whole cloud moves", "It assumes minimal movement"],
        ["Hard or soft", "Entropic (Sinkhorn-Knopp), 5 partners with weights", "Exact OT, one partner each", "One-to-one matches claim a certainty the data does not have; the entropic solve is also fast", "Blur: 60 to 75 controls per cell"],
        ["Size of the blur ε", "0.05 × median squared distance between controls", "Scale by the control-to-perturbed distance", "Otherwise strong perturbations get blurrier pairs simply because they moved further. Changed before any real-data run", "One more constant, not tuned"],
        ["Problem size", "One per perturbation and gem group", "One per perturbation", "Pairs never cross sequencing batches", "Small problems: as few as 5 perturbed cells in training, 1 in validation"],
        ["When the label must stay hidden", "A second, pooled pairing: controls against all single-perturbed cells at once", "Reuse the per-perturbation pairs", "Per-perturbation pairs already use the label, which the identification test must not know", "Twice the pair files"]])}
    {code("src/pertact/ot.py", "def sinkhorn(C, eps", "return P, {", "sinkhorn: the call to the POT library's solver", True)}
    {code("src/pertact/ot.py", "def pair(Xc, Xp", "return top, w, info", "pair: cost, blur, top-5 partners")}
  </div>''')

# ---- 10 actions
P(f'''
  <div class="block" id="wk-actions">
    <div class="kicker">10 · Week 2: action vectors</div>
    <h2>Turning pairs into one vector per gene</h2>
    <p class="lede">Two methods read the same OT pairs and output the same thing: a table F with one row per gene, <span class="mono">[105, 50]</span> in PCA-50 or <span class="mono">[105, 30]</span> in scVI-30. The mean shift averages. InfoNCE trains.</p>
    {fig(F.fig_infonce(), "<b>One InfoNCE step with its tensors, and the mean shift below it.</b> In the logits matrix the dark diagonal holds each query's own OT partner. The pink blocks are other cells of the same perturbation, which share f(a) and can only be told apart by the control's own state. F is the whole model: 5,250 numbers in PCA-50.")}
    <ul class="clean">
      <li><b>What the loss asks.</b> Move a control by its gene's action. The result should land closer to that control's own partner than to any other perturbed cell in the batch.</li>
      <li><b>Two kinds of wrong answer.</b> Cells of other genes are easy to avoid if f(a) points the right way. The 7 other cells of the same gene are difficult: they got the same f(a).</li>
      <li><b>Temperature τ</b> sets how sharply near misses are punished. It is in units of D₀, the average squared distance between two random training cells (120.1 and 33.1), so one grid serves both spaces. Chosen on validation: 0.03 and 0.1.</li>
      <li><b>Cost.</b> Training takes seconds: 1,400 to 3,800 steps, stopped on validation loss.</li>
    </ul>
    {trade([
        ["What an action is", "One free vector per gene", "A network from gene features, or from the cell's state", "It is the simplest form, and it compares one-to-one with the mean shift, so only the objective differs", "No vector for an unseen gene; the same shift for every cell"],
        ["Score", "Negative squared distance", "Cosine similarity", "The model claims control + f(a) lands on the cell in real coordinates; cosine throws length away", "Length still came out wrong (section 11)"],
        ["Positives", "Only the cell's own OT partner", "Every cell of the same gene", "Same-gene positives collapse to a class average, which is the mean shift again, and never test state + action cell by cell", "A noisy target: the partner is itself a guess"],
        ["Batch", "32 genes × 8 pairs", "Random pairs", "Guarantees 7 difficult negatives per query", "Rare genes are sampled as often as common ones"]])}
    <p><b>The reference.</b> The mean shift is the exact least-squares answer for one fixed vector per gene. It needs no training, and it is the baseline InfoNCE has to beat. Week 2 → Method, part 4 sets the two side by side.</p>
    {code("src/pertact/infonce.py", "def infonce_loss", "return loss", "infonce_loss: the whole objective", True)}
    {code("src/pertact/infonce.py", "def batch(self, rng, P, K)", "torch.as_tensor(self.a[rows]))", "PairData.batch: 32 genes × 8 pairs, partner drawn from the OT weights")}
    {code("src/pertact/infonce.py", "F = torch.zeros(len(genes)", "break", "the training loop")}
    {code("src/pertact/actions.py", "def mean_shift_actions", "return {g: m.loc[g]", "mean_shift_actions: the closed-form reference")}
  </div>''')

# ---- 11 tests
P(f'''
  <div class="block" id="wk-tests">
    <div class="kicker">11 · Week 2: tests and the length problem</div>
    <h2>Three tests on held-out cells, and why length got in the way</h2>
    <p class="lede">All three use validation cells: perturbations seen in training, cells that were not. None uses a double.</p>
    {fig(F.fig_tests(), "<b>What each test asks of an action.</b> The first is the training task on new cells. The second hides the cell's label and asks which of 105 actions explains it. The third reuses the Week 1 decoder with the action as its gene vector.")}
    {table(["Test (PCA-50 / scVI-30)", "InfoNCE as trained", "InfoNCE at mean-shift lengths", "Mean shift"], [
        ["Find the cell (MRR; 1.0 = always ranked first)", "<b>0.181 / 0.174</b>", "0.114 / 0.125", "0.120 / 0.133"],
        ["Name the perturbation (top-1; chance 0.009)", "0.095 / 0.093", "0.255 / 0.187", "0.235 / 0.191"],
        ["Decoder, every vector cut to length 1", "0.959 / 0.965", "same vectors", "0.961 / 0.962"]])}
    <ul class="clean">
      <li><b>InfoNCE vectors come out too long:</b> 2.1× and 2.4× the mean shift at the median, about 4× for weak perturbations. The loss rewards landing far from other cells, and nothing in it pins the length.</li>
      <li><b>Too long helps one test and breaks another.</b> It wins "find the cell" and fails "name the perturbation", where the overshoot lands among another gene's cells.</li>
      <li><b>The win is length, not direction.</b> Give InfoNCE's directions the mean shift's lengths and they score no better than the mean shift on either test.</li>
      <li><b>Part of "find the cell" comes from the pairing.</b> The starting control was itself chosen by pairing it with the held-out cell. With no action at all the score is 0.026 / 0.031, against 0.001 for a random ranking, so actions are judged by how far they rise above that.</li>
    </ul>
    <h3>Ways to fix the length</h3>
    {table(["Fix", "What it is", "What it buys", "What it costs"], [
        ["<b>One global scale α</b>", "One number for all actions, fitted by least squares on training pairs (α ≈ 0.42)", "Training is untouched; the median length comes to 0.88 and 1.00", "It keeps InfoNCE's even lengths, so weak genes stay 1.6 to 1.8× long. Retrieval falls below the mean shift"],
        ["<b>One scale per gene</b>", "The same least squares, gene by gene. In closed form it is the projection of the mean shift onto f(a)", "Best at naming the perturbation (0.268 in PCA-50)", "It always undershoots, by the cosine between the two (median 0.80 and 0.85). Retrieval stays below the mean shift"],
        ["<b>Anchor term in training</b>", "A squared-error pull toward the partner, added to the loss", "Lengths of 1.18× and 1.30× without any rescaling", "It only works when it is 59 to 73% of the loss, and by then it is the reconstruction objective. Flagged and not carried on"],
        ["<b>Unit length</b>", "Every action divided by its own length; for decoder input only", "Removes the gap to one-hot on weak perturbations", "It throws the length away, so it cannot be used for adding actions"]])}
    <p><b>Where that leaves the action:</b> keep it as a direction plus a length fitted on training pairs. Give the decoder the direction. Use direction × length when adding actions in the state space.</p>
    {code("src/pertact/actions.py", "def per_action_scale", "for g in tot.index if g in actions}", "per_action_scale: the per-gene length")}
    {code("src/pertact/retrieval.py", "def identification", '"mrr": float((1 / r).mean())}, per', "identification: try every action from a label-free start")}
  </div>''')

# ---- 12 status
P('''
  <div class="block" id="wk-status">
    <div class="kicker">12 · Where it stands</div>
    <h2>What is established, what is not, and the weak points</h2>
    <div class="callout key">
      <h3>Established</h3>
      <ul class="clean">
        <li>The data pipeline, split and scoring are in place, with nothing fitted on validation or test cells.</li>
        <li>With borrowed gene vectors, no decoder beats adding the two singles on held-out doubles, and none captures interactions.</li>
        <li>Entropic OT gives pairs that pass every check available without ground truth.</li>
        <li>InfoNCE actions are 2 to 4× too long. Once lengths are matched they tie the plain mean shift on all three tests.</li>
      </ul>
    </div>
    <div class="callout">
      <h3>Not tested yet</h3>
      <ul class="clean">
        <li><b>Composition with actions.</b> Does f(a) + f(b) land on held-out doubles? This is the test that can separate InfoNCE from the mean shift, and it will be run once on the test set.</li>
        <li><b>Unseen genes.</b> Needs an action computed from gene features instead of looked up.</li>
        <li><b>Actions that depend on the starting cell.</b></li>
      </ul>
    </div>
    <div class="callout">
      <h3>Weak points to keep in mind</h3>
      <ul class="clean">
        <li>Every Week 2 number is a validation number, and validation was used to choose settings.</li>
        <li>OT pairs rest on an assumption that cannot be checked directly.</li>
        <li>Scores on average profiles cannot see cell-to-cell differences, which is where a cell's starting state would matter.</li>
        <li>One dataset whose doubles are mostly additive. A method could fail to beat addition here and still be useful elsewhere.</li>
      </ul>
    </div>
  </div>
</div>
</section>
''')

WALK = "\n".join(parts)

CSS = '''
/* Walkthrough tab */
#w-walk:checked ~ main #walk { display: block; }
#w-walk:checked ~ nav.weeks label[for="w-walk"] { color: var(--ink); border-bottom-color: var(--ink); }
#w-walk:focus-visible ~ nav.weeks label[for="w-walk"] { outline: 2px solid var(--accent); outline-offset: 2px; }
.walk { padding-block: 28px 0; display: grid; gap: 44px; }
.walk > * { min-width: 0; }
.walk > .block { scroll-margin-top: 12px; gap: 14px; }
.walk .kicker { font: 500 var(--step--1)/1.4 var(--f-mono); letter-spacing: .06em; text-transform: uppercase; color: var(--accent); }
.walk h3 { margin-top: 6px; }
.walk ol.steps h3, .walk .callout h3 { margin-top: 0; }
.walk-toc { margin: 0; padding: 14px 16px 14px 40px; display: grid; grid-template-columns: repeat(auto-fit, minmax(min(240px, 100%), 1fr)); gap: 4px 28px; font-size: var(--step--1); background: var(--card); border: 1px solid var(--rule); border-radius: 8px; }
.walk-toc a { color: var(--ink); text-decoration: none; border-bottom: 1px solid var(--rule); }
.walk-toc a:hover { border-bottom-color: var(--accent); }
.trade table { min-width: 820px; }
.trade td:first-child { font-weight: 600; }
.trade td { min-width: 110px; }
details.code { background: var(--card); border: 1px solid var(--rule); border-radius: 8px; min-width: 0; }
details.code summary { cursor: pointer; padding: 10px 14px; font-size: var(--step--1); color: var(--muted); }
details.code summary .mono { color: var(--ink); }
details.code[open] summary { border-bottom: 1px solid var(--rule); }
details.code pre { border-radius: 0 0 8px 8px; max-height: 560px; overflow: auto; }
.lk { display: inline-block; font: 600 11.5px/1 var(--f-body); padding: 4px 7px; border-radius: 4px; white-space: nowrap; }
.lk.fit { color: var(--ok); background: var(--ok-soft); } .lk.app { color: var(--muted); background: var(--code-bg); }
.lk.pick { color: var(--warn); background: var(--warn-soft); } .lk.score { color: var(--accent); background: var(--accent-soft); }
.lk.no { color: var(--muted); font-weight: 400; padding-left: 0; }
.d-m { fill: var(--ink); font: 500 12px var(--f-mono); }
.d-h { fill: var(--ink); font: 600 12.5px var(--f-body); }
.d-xs { fill: var(--muted); font: 400 11.5px var(--f-body); }
.d-ok { fill: var(--ok); font: 600 11px var(--f-body); }
.d-w { fill: #fff; font: 600 12.5px var(--f-body); } .d-wm { fill: #fff; font: 400 12.5px var(--f-mono); }
.d-ten { fill: var(--accent-soft); stroke: var(--accent); stroke-width: 1.2; }
.d-ten.ctl { fill: var(--code-bg); stroke: var(--muted); }
.d-ten.strong { stroke-width: 2.6; }
.d-op { fill: var(--card); stroke: var(--ink); stroke-width: 1.2; }
.d-out { fill: none; stroke: var(--dbl); stroke-width: 1.2; stroke-dasharray: 5 4; }
.d-seg { fill: var(--code-bg); stroke: var(--rule); stroke-width: 1; }
.d-seg.hot { fill: var(--accent-soft); stroke: var(--accent); } .d-seg.code { fill: var(--accent); stroke: var(--accent); }
.d-box.none { fill: none; stroke-dasharray: 4 4; }
.d-fill-acc { fill: var(--accent); opacity: .3; } .d-fill-ctl { fill: var(--muted); opacity: .26; } .d-fill-dbl { fill: var(--dbl); opacity: .3; }
'''


def patch():
    s = SITE.read_text()

    def rep(old, new, count=1):
        nonlocal s
        assert s.count(old) == count, (s.count(old), old[:70])
        s = s.replace(old, new)

    # idempotent: drop an earlier walkthrough before inserting
    s = re.sub(r"\n/\* Walkthrough tab \*/.*?(?=</style>)", "", s, flags=re.S)
    s = re.sub(r'<section class="week" id="walk".*?</section>\n', "", s, flags=re.S)
    s = s.replace('<input type="radio" name="week" class="tabradio" id="w-walk" value="walk" aria-label="Walkthrough">\n', "")
    s = s.replace('    <label for="w-walk">Walkthrough<small>start here</small></label>\n', "")
    s = re.sub(r'\n    <p class="lede" id="home-walk">.*?</p>', "", s, flags=re.S)
    s = s.replace('      if (/^wk-/.test(h)) { document.getElementById("w-walk").checked = true; return; }\n', "")

    rep("</style>", CSS + "</style>")
    rep('<input type="radio" name="week" class="tabradio" id="w-home" value="home" checked aria-label="Home">\n',
        '<input type="radio" name="week" class="tabradio" id="w-home" value="home" checked aria-label="Home">\n'
        '<input type="radio" name="week" class="tabradio" id="w-walk" value="walk" aria-label="Walkthrough">\n')
    rep('    <label for="w-home">Home</label>\n', '    <label for="w-home">Home</label>\n    <label for="w-walk">Walkthrough<small>start here</small></label>\n')
    rep('<section class="week" id="week1" aria-label="Week 1">', WALK + '<section class="week" id="week1" aria-label="Week 1">')
    rep('      if (w) { w.checked = true; return; }\n',
        '      if (w) { w.checked = true; return; }\n      if (/^wk-/.test(h)) { document.getElementById("w-walk").checked = true; return; }\n')
    # pointer on Home, under the hero question
    m = re.search(r'(<div class="hero">\s*<div class="eyebrow">The question</div>.*?</p>\s*<p class="lede">.*?</p>)', s, flags=re.S)
    assert m
    s = s[:m.end()] + ('\n    <p class="lede" id="home-walk">New here, or want the code and tensors in one place? '
                       '<label class="open" for="w-walk" style="display:inline-block;margin-left:6px;font:600 var(--step--1)/1 var(--f-body);color:var(--accent);'
                       'border:1px solid var(--accent);border-radius:6px;padding:7px 11px;cursor:pointer">Open the Walkthrough</label></p>') + s[m.end():]
    SITE.write_text(s)
    print("walkthrough:", len(WALK), "chars; site:", len(s))


patch()
