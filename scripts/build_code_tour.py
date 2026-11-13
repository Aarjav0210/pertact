"""Build the Code tab of site/index.html: every source file in full, cut into functions, each with a note.

The source is read at build time, so the listings always match the files. The notes and the review list are
written by hand in this script, keyed by file and function name. A function without a note is printed as a warning,
and a note for a function that no longer exists stops the build, so the notes cannot drift silently.

    python -I scripts/build_code_tour.py

Idempotent: an earlier Code tab is replaced. Safe to run before or after scripts/build_walkthrough.py.
"""
import ast
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site" / "index.html"
REVIEWED = "11 November 2026"

try:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import BashLexer, PythonLexer, TOMLLexer, YamlLexer
    LEXERS = {".py": PythonLexer, ".yaml": YamlLexer, ".sh": BashLexer, ".toml": TOMLLexer}
except ImportError:  # the page still builds, without colours
    highlight = None


# ------------------------------------------------------------------ source -> segments
def read_lines(rel):
    return (ROOT / rel).read_text().split("\n")


def html_lines(rel, lines):
    """One HTML string per source line, syntax-coloured when pygments is installed."""
    if highlight is None:
        return [html.escape(l) for l in lines]
    lexer = LEXERS[Path(rel).suffix](stripnl=False, stripall=False, ensurenl=False)
    out = highlight("\n".join(lines), lexer, HtmlFormatter(nowrap=True, classprefix="p-")).split("\n")
    out = out[:len(lines)] + [""] * (len(lines) - len(out))
    assert len(out) == len(lines), (rel, len(out), len(lines))
    assert all(l.count("<span") == l.count("</span>") for l in out), f"{rel}: a colour span crosses a line"
    return out


def segments(rel, lines):
    """Cut a file into (key, label, first line, last line), 1-based and inclusive. Python files are cut at every
    top-level function and class; statements between them form their own segments. Every non-blank line of the
    file lands in exactly one segment."""
    if not rel.endswith(".py"):
        return [("@file", "whole file", 1, len(lines))]
    items, group = [], None
    for node in ast.parse("\n".join(lines)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if group:
                items.append(group)
                group = None
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            items.append([node.name, f"{kind} {node.name}", node.end_lineno])
        else:
            is_main = isinstance(node, ast.If) and "__name__" in ast.dump(node.test)
            name = next((t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)), None)
            if is_main:
                if group:
                    items.append(group)
                group = ["@main", "entry point", node.end_lineno]
            elif group is None:
                first = not items
                group = ["@head" if first else f"@{name or 'code'}", "module header" if first else (name or "module code"),
                         node.end_lineno]
            else:
                group[2] = node.end_lineno
    if group:
        items.append(group)
    segs, prev, seen = [], 0, {}
    for i, (key, label, end) in enumerate(items):
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:  # two unnamed blocks of module code in one file
            key = f"{key}{seen[key]}"
        a, b = prev + 1, (len(lines) if i == len(items) - 1 else end)
        while a < b and not lines[a - 1].strip():
            a += 1
        while b > a and not lines[b - 1].strip():
            b -= 1
        segs.append((key, label, a, b))
        prev = end
    covered = {n for _, _, a, b in segs for n in range(a, b + 1)}
    assert all((n in covered) or not l.strip() for n, l in enumerate(lines, 1)), f"{rel}: a line is in no segment"
    return segs


def ref(rel, needle, exact=False):
    """'file:line' of the first line containing `needle`, so review notes keep pointing at the right line."""
    for n, l in enumerate(read_lines(rel), 1):
        if (l.strip() == needle.strip()) if exact else (needle in l):
            return f'<a href="#{slug(rel)}"><code>{rel}:{n}</code></a>'
    raise AssertionError(f"{needle!r} not found in {rel}; update the review notes in build_code_tour.py")


def slug(rel):
    return "ct-" + re.sub(r"[^a-z0-9]+", "-", rel.lower()).strip("-")


def md(s):
    """Notes are plain text with `code` in backticks."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(s, quote=False))


# ------------------------------------------------------------------ the files, in run order
# cells: which sets of cells the file reads. TR = training, VA = validation, TE = test (the double perturbations).
GROUPS = [
    ("Shared foundations", "Settings, data preparation, the split and the two state encoders. Every later step reads what these write.", [
        dict(path="src/pertact/__init__.py", role="Package marker. Sets one environment variable before anything else loads.",
             cells="none"),
        dict(path="src/pertact/config.py", role="Loads `config.yaml`, resolves paths, seeds the random number generators, and holds the small JSON and logging helpers every module uses.",
             cells="none"),
        dict(path="config.yaml", role="Every setting in one place: file paths, cell filters, split fractions, encoder sizes, embedding settings, OT, InfoNCE, decoder and evaluation settings.",
             cells="none"),
        dict(path="src/pertact/data.py", role="Reads the raw Norman 2019 file, works out which perturbation each cell received, filters cells, and builds the model matrix of 5,028 genes.",
             run=["python -I -m pertact.data qc", "python -I -m pertact.data hvg"],
             reads="`data/raw/` (scPerturb file, HGNC table), `splits.json` for stage 2",
             writes="`data/interim/norman2019_qc_allgenes.h5ad`, `data/processed/norman2019.h5ad`, QC statistics",
             cells="Stage 1: all cells, cell-level filters only. Stage 2: gene choices from TR cells only."),
        dict(path="src/pertact/splits.py", role="Assigns every cell to training, validation, test or excluded, once, with a fixed seed.",
             run=["python -I -m pertact.splits"], reads="the QC'd file's cell labels", writes="`splits.json`",
             cells="labels of all cells"),
        dict(path="src/pertact/encoders.py", role="Fits the two frozen state encoders (PCA-50, scVI-10 and scVI-30) on training cells and applies them to every cell. Also the loader every later step starts from.",
             run=["python -I -m pertact.encoders pca", "python -I -m pertact.encoders scvi", "python -I -m pertact.encoders umap",
                  "python -I -m pertact.encoders write_h5ad"],
             reads="`data/processed/norman2019.h5ad`", writes="`data/processed/state_embeddings.npz`, `models/pca_g.npz`, `models/scvi10/`, `models/scvi30/`, three UMAP figures",
             cells="Fitted on TR cells. Applied to all cells. The UMAP figures draw all cells, for display only."),
    ]),
    ("Week 1: gene embeddings, decoders and scoring", "Four ways to describe a gene as a vector, one decoder to test them all, and the scoring on validation cells and test cells.", [
        dict(path="src/pertact/embeddings.py", role="Builds the four Week 1 perturbation embeddings: one-hot, PCA gene loadings, GenePT and STRING node2vec. Also builds GenePT and STRING vectors for every measured gene.",
             run=["python -I -m pertact.embeddings", "python -I -m pertact.embeddings all_genes"],
             reads="processed data, `splits.json`, the GenePT, STRING and HGNC downloads", writes="`embeddings/<method>.npz`, `embeddings/coverage.json`, all-gene tables",
             cells="PCA loadings: TR cells of the singles. The other three use no cells."),
        dict(path="src/pertact/decoders.py", role="The evaluation model (latent additive and decoder-only), its training loop and its prediction functions. A library with no command of its own.",
             cells="whatever the caller passes in"),
        dict(path="src/pertact/metrics.py", role="Scoring: the most changed genes of each perturbation, the fixed references of a split, every metric, and the noise ceiling. A library.",
             writes="`results/de_genes.json`", cells="DE genes: TR cells for singles, a double's own TE cells for doubles. References: the scored split."),
        dict(path="src/pertact/runs.py", role="The Week 1 experiment: the baselines, decoder tuning, the runs over embeddings, decoders and seeds, the sequential-composition test, and rescoring from saved predictions.",
             reads="processed data, states, embeddings, `splits.json`", writes="`results/week1*.csv`, `results/predictions/`, `models/decoders/`, `results/best_hparams.json`",
             cells="Trains on TR. Early stopping and tuning on VA. Scores VA and TE."),
        dict(path="src/pertact/evaluate.py", role="The command line for the scoring, the Week 1 runs and the Week 2 decoder test.",
             run=["python -I -m pertact.evaluate de", "python -I -m pertact.evaluate baselines", "python -I -m pertact.evaluate tune",
                  "python -I -m pertact.evaluate run", "python -I -m pertact.evaluate compose", "python -I -m pertact.evaluate rescore",
                  "python -I -m pertact.evaluate actions --variants ... --tag ..."],
             cells="see the three modules it calls"),
    ]),
    ("Week 2: pairing, action vectors and their tests", "The three stages we went through together: pairing with OT, training action vectors, and testing them on validation cells.", [
        dict(path="src/pertact/ot.py", role="Pairs each perturbed cell with control cells by entropic optimal transport, using the POT library's Sinkhorn-Knopp solver: per perturbation and, without labels, pooled.",
             run=["python -I -m pertact.ot --check", "python -I -m pertact.ot"],
             reads="states, `splits.json`", writes="`results/ot/<space>__<split>.npz`, the `__pooled` versions, `diagnostics.csv`, `checks.json`",
             cells="TR cells paired with TR controls, VA cells with VA controls. No TE cell."),
        dict(path="src/pertact/actions.py", role="Sets of action vectors: saving and loading them, the mean-shift reference, the two rescalings, and length comparisons. A library.",
             reads="states, per-perturbation OT pairs of TR cells", cells="TR pairs only"),
        dict(path="src/pertact/infonce.py", role="Learns one action vector per gene with the InfoNCE loss on the OT pairs. Also tunes the temperature and runs the length-anchor variant.",
             run=["python -I -m pertact.infonce tune", "python -I -m pertact.infonce train", "python -I -m pertact.infonce anchor"],
             reads="states, per-perturbation OT pairs", writes="`embeddings/infonce__<space>__s<seed>.npz`, `embeddings/infonce_anchor__...`, `results/infonce/`",
             cells="Fits on TR pairs. Early stopping and temperature choice on VA pairs. No TE cell."),
        dict(path="src/pertact/retrieval.py", role="The forward and reverse tests of any set of action vectors, run on every set.",
             run=["python -I -m pertact.retrieval"], reads="states, OT pairs (per-perturbation and pooled), action vectors",
             writes="`results/week2_retrieval.csv`, `week2_identification_by_pert.csv`, `week2_length_ratio.csv`, `week2_action_scales.csv`, `week2_effect_strength.csv`",
             cells="Tests on VA cells, and on TR cells for comparison. No TE cell."),
        dict(path="src/pertact/decoder_test.py", role="The decoder test: frozen action vectors as the input of the Week 1 decoders. Scored on validation cells only, or with `--doubles` also on the test doubles.",
             run=["python -I -m pertact.evaluate actions --variants ... --tag ...", "python -I -m pertact.evaluate actions --doubles --variants ... --tag ..."],
             writes="`results/week2_decoder/`; with `--doubles`, `results/week3_decoder/`",
             cells="Trains on TR, early stopping on VA, scores VA. Built without the test split unless `--doubles` is given; then TE cells are scored too, and only scored."),
    ]),
    ("Week 3: adding two action vectors, scored on the test doubles", "One file. It fits nothing: it takes the Week 2 action vectors as they are and scores f(a) + f(b) against the real doubles, in the state space.", [
        dict(path="src/pertact/doubles.py", role="Scores every Week 2 set of action vectors on the 127 test doubles: how close the sum of two vectors lands to the double's centre, and whether the double can be named from a cell.",
             run=["python -I -m pertact.doubles --split val", "python -I -m pertact.doubles"], reads="states, `splits.json`, OT pairs of TR cells, action vectors",
             writes="`results/week3_doubles.csv`, `week3_doubles_by_pair.csv`",
             cells="Scores TE cells: the doubles and the test controls. With `--split val`, VA cells instead. Fits nothing on either."),
    ]),
    ("Analysis scripts", "Small scripts that turn saved results into tables, diagnostics and one figure. None of them trains anything.", [
        dict(path="scripts/perturbation_map.py", role="Describes the experiment's design for the site's Data tab: which singles and doubles exist, example cells, and a few condition averages.",
             run=["python -I scripts/perturbation_map.py"], writes="`results/perturbation_map.json`", cells="all cells, for display only"),
        dict(path="scripts/report_tables.py", role="Prints the Week 1 results tables used in `REPORT.md`.", run=["python -I scripts/report_tables.py"],
             reads="`results/week1.csv`", cells="none (reads saved scores)"),
        dict(path="scripts/week2_direction_diagnostics.py", role="Measures how spread out each set of action vectors is, and how their lengths differ between weak and strong perturbations.",
             run=["python -I scripts/week2_direction_diagnostics.py"], writes="`results/week2_decoder/direction_spread.csv`, `length_overlap.csv`",
             cells="TR pairs only (through the action vectors)"),
        dict(path="scripts/week2_decoder_summary.py", role="Summarises the Week 2 decoder test by seed and by effect-strength third.",
             run=["python -I scripts/week2_decoder_summary.py"], reads="`results/week2_decoder/smoke_per_perturbation__*.csv`",
             writes="`results/week2_decoder/summary.csv`, `paired_length_vs_unit.csv`", cells="none (reads saved VA scores)"),
        dict(path="scripts/week3_week1_in_state_space.py", role="Scores the saved Week 1 predictions of the test doubles with the Week 3 state-space scores, by projecting them into PCA-50. Not like for like: these rows have a trained decoder, the action vectors were added directly.",
             run=["python -I scripts/week3_week1_in_state_space.py"], reads="`results/predictions/`, `results/week3_doubles_by_pair.csv`, `models/pca_g.npz`",
             writes="`results/week3_doubles_week1_embeddings.csv`, `week3_doubles_week1_embeddings_by_pair.csv`", cells="TE cells: the doubles and the test controls, to score saved predictions. Fits nothing."),
        dict(path="scripts/plot_ot_pairs.py", role="Draws the OT pairs of a few perturbations in a 2-D slice of a state space, next to random pairs.",
             run=["python -I scripts/plot_ot_pairs.py"], writes="`figures/ot_pairs_X_pca_g.png`, `results/ot/pair_plot_stats_X_pca_g.csv`", cells="TR cells of one gem group"),
    ]),
    ("Checks and tests", "How to confirm that the code still produces the saved results, and that its building blocks behave.", [
        dict(path="scripts/check_outputs.py", role="Reruns the cheap, deterministic steps into temporary folders and compares every file with the saved one.",
             run=["python -I scripts/check_outputs.py", "python -I scripts/check_outputs.py --steps ot"], cells="TR and VA cells; TE cells only where the rerun step scores them"),
        dict(path="tests/test_data_and_splits.py", role="Guide parsing, perturbation labels and the split, on tiny hand-made inputs.", run=["python -I -m pytest"], cells="none (synthetic)"),
        dict(path="tests/test_ot_and_tests.py", role="The OT pairing, the start state and the rank used by both tests, on synthetic clouds.", cells="none (synthetic)"),
        dict(path="tests/test_doubles.py", role="The scores of the doubles test, on synthetic cells with one additive and one non-additive double.", cells="none (synthetic)"),
        dict(path="tests/test_models_and_scores.py", role="The embedding table, two structural properties of the decoder, the InfoNCE loss and the scoring helpers.", cells="none (synthetic)"),
    ]),
    ("Site builders and packaging", "Tooling for this notebook. None of it affects a result.", [
        dict(path="scripts/build_site.py", role="Injects the Week 1 scores and the perturbation map into this page as JSON.", run=["python -I scripts/build_site.py"],
             cells="none", closed=True),
        dict(path="scripts/build_walkthrough.py", role="Builds the Walkthrough tab: prose, tables and code excerpts, and patches it into this page.",
             run=["python -I scripts/build_walkthrough.py"], cells="none", closed=True),
        dict(path="scripts/walkthrough_figs.py", role="Draws the Walkthrough's SVG figures. One function per figure.", cells="none", closed=True),
        dict(path="scripts/build_code_tour.py", role="Builds this Code tab.", run=["python -I scripts/build_code_tour.py"], cells="none", closed=True),
        dict(path="scripts/wrap_preview.sh", role="Wraps the page so it opens straight from disk as `site/notebook.html`.", run=["sh scripts/wrap_preview.sh"],
             cells="none", closed=True),
        dict(path="pyproject.toml", role="Makes `src/pertact` installable and lists the packages it imports.", cells="none", closed=True),
    ]),
]

# ------------------------------------------------------------------ notes: file -> segment key -> text
# "note || review remark". Keys are function or class names; "@head" is the top of a file, "@main" its last lines.
N = {}
N["src/pertact/__init__.py"] = {
    "@head": "numba (used by scanpy, UMAP and PecanPy) crashes on macOS when PyTorch's threading library is loaded in the same process. Choosing numba's `workqueue` layer avoids that. It has to be set before numba is imported, so it lives here.",
}
N["src/pertact/config.py"] = {
    "@head": "`ROOT` is the repository folder, found from this file's own location. Every relative path in `config.yaml` is resolved against it, so commands work from any directory. `VAL_SEED_OFFSET` keeps the random draws used for validation apart from the training draws.",
    "load_config": "Reads the config into a plain dictionary: the file given with `--config`, else the file named by the `PERTACT_CONFIG` environment variable, else `config.yaml`.",
    "path": "Turns a name under `paths:` in the config into a full path. Every module finds its files this way, so moving a file means changing one line of the config.",
    "seed_everything": "Seeds Python, NumPy and, if installed, PyTorch. Called at the start of each command and again before each training run.",
    "get_logger": "One log format for every module.",
    "write_json": "Writes JSON, creating the folder if needed. NumPy numbers are converted to plain ones.",
    "read_json": "Reads a JSON file.",
}
N["config.yaml"] = {
    "@file": "Read top to bottom, it follows the pipeline: paths, cell filters, the split, encoders, Week 1 embeddings, then Week 2 (`ot`, `infonce`), then the decoder and evaluation settings. The decoder block holds defaults; the tuned values in `results/best_hparams.json` override them at run time. || `ot.k` (5 partners) and `ot.eps_scale` (0.05) were starting values and have not been tuned. The 5 heaviest controls hold a median 23% of a cell's OT weight.",
}
N["src/pertact/data.py"] = {
    "@head": "The docstring gives the order: cell-level filtering first, then the split, then gene selection. Gene selection comes after the split so that it can use training cells only. `CONTROL` is the label every unperturbed cell gets.",
    "construct_genes": "Reads one guide construct name and returns its target genes. Control tokens (`NegCtrl...`) and numeric suffixes are dropped, and the genes are sorted, so `A_B` and `B_A` become the same perturbation.",
    "parse_guide_id": "A cell's guide label can list several constructs. The cell is kept only if they all name the same genes, no gene appears twice, and there are at most two genes.",
    "condition_name": "Gene tuple to label: no genes is `control`, one gene is `KLF1`, two are `CEBPE+RUNX1T1`.",
    "condition_genes": "The reverse: label to list of genes. Used everywhere a double has to be split into its two singles.",
    "SymbolResolver": "Loads the HGNC table so that old or alternative gene names can be mapped to current ones and to Ensembl ids. Only unambiguous matches are accepted.",
    "resolve_targets_to_var": "Finds the column of the expression matrix for each perturbed gene: by exact name, then by Ensembl id, then by synonym. It records how each match was made, so a wrong match can be traced.",
    "annotate": "For every cell: its perturbation, how many genes were targeted, and, if it must be dropped, why (low guide coverage, more than one guide identity, or an ambiguous guide).",
    "cross_check_labels": "Compares the labels parsed here with the labels scPerturb ships, and counts disagreements. The counts go into the QC statistics.",
    "build_qc": "Stage 1. Reads the raw file and applies, in order: the guide filter, cell quality filters (at least 2,000 counts and 1,000 genes, at most 15% mitochondrial), and the rule that a perturbation needs 50 cells. It keeps all genes and raw counts. No gene is chosen here, which is why this stage can run before the split exists.",
    "summarise_conditions": "Counts cells per perturbation for the statistics file: how many singles and doubles, and the smallest, median and largest group.",
    "build_processed": "Stage 2. Labels each cell with its set, then makes the gene choices from training cells only: genes seen in at least 20 training cells, and the 5,000 most variable genes among them. Every perturbed gene is added even if it was not selected, which gives 5,028. Normalisation (10,000 counts per cell, then log) is done cell by cell, so it cannot leak.",
    "main": "Chooses the stage from the command line.",
    "@main": "Runs `main` when the module is called as a command.",
    "cell_labels": "Each cell's perturbation, gem group and split, as three plain arrays. The pairing, the reverse test and two scripts all start from these.",
}
N["src/pertact/splits.py"] = {
    "@head": "The docstring states the rule for each set.",
    "make_splits": "For each single perturbation, its cells are shuffled and 20% go to validation. Controls are split 70 / 15 / 15 inside each gem group, so every set has controls from every batch. Test is every double whose two genes both exist as singles. Two assertions close the function: every cell is in exactly one set, and no double-perturbation cell is in training or validation.",
    "main": "Builds the split from the QC'd file's labels and writes it.",
    "@main": "Runs `main`.",
}
N["src/pertact/encoders.py"] = {
    "@head": "A state encoder turns a cell's 5,028 expression values into a short vector. Both encoders here are fitted on training cells, then frozen.",
    "train_mask": "True for training cells.",
    "state_store": "Where the state vectors are kept: one file next to the model matrix.",
    "save_state": "Adds new state arrays to that file. It first checks that the cells are in the same order as before.",
    "attach_state": "Loads the stored states into the data object, again checking the cell order. Every later module gets its states through this.",
    "fit_pca": "Fits a 50-component PCA on the training cells' log-normalised expression, then projects every cell with those fixed loadings. The mean and loadings are saved, because the sequential-composition test needs to re-encode a predicted cell with the same PCA.",
    "fit_scvi": "Trains scVI from scratch, once per latent size (10 and 30), on the raw counts of training cells, with the gem group as the batch. scVI's own early stopping uses a validation share it draws from those training cells. The trained model then gives every cell its latent mean. No public pretrained weights are involved.",
    "umap_figures": "Three sanity-check pictures per state space: control against perturbed, four strong perturbations, and gem group. || The UMAP is fitted on a sample of all cells, test cells included. It is used only for these pictures.",
    "main": "Runs one of the four steps. `write_h5ad` copies the stored states into the model matrix file.",
    "@main": "Runs `main`.",
    "load_all": "The model matrix with every state space attached, plus the split. Every later module starts from this.",
    "states": "The state vectors of one space, one row per cell.",
    "scale_D0": "The average squared distance between two random training cells. Dividing by it puts the temperature on the same scale in both state spaces.",
}
N["src/pertact/embeddings.py"] = {
    "@head": "A perturbation embedding is one vector per gene. Each method writes the same three arrays: the vectors, the genes they cover, and the genes it could not cover. Missing genes are never filled in.",
    "emb_path": "File name for a method's embedding.",
    "save_embedding": "Writes one embedding and updates `coverage.json` with its size, what is missing and how names were matched.",
    "load_embedding": "Reads one embedding back as (genes, matrix).",
    "all_targets": "Every gene that is perturbed anywhere in the data, alone or in a double.",
    "build_onehot": "The identity matrix over the 105 training genes: each gene gets its own axis and nothing is shared between genes.",
    "build_pca_loadings": "Averages each single perturbation's training cells into one expression profile, stacks them (perturbations by genes), and takes the top 32 directions of that table. A gene's vector is its row of loadings, so two genes are close if their expression moves together across perturbations.",
    "load_genept": "Returns GenePT's text-derived vector for every gene symbol in the public file. The first call builds a compact cache, because the original unpacks to several gigabytes.",
    "_stream_genept_pickle": "Reads the pickle one instruction at a time and collects the numbers, without ever executing it. This avoids running code from a downloaded file, and keeps memory small.",
    "match_symbols": "Matches each perturbed gene to a name in an outside vocabulary, trying the guide's name, the matrix name, the current HGNC name and its synonyms, in that order.",
    "build_genept": "Looks up the GenePT vector of each perturbed gene (1,536 numbers).",
    "string_graph": "Builds the STRING protein network, keeping links with a combined score of at least 700.",
    "string_symbol_maps": "Gene name to STRING protein id, by preferred name and by unambiguous alias.",
    "node2vec_embed": "Runs node2vec on the network: random walks, then a skip-gram model that gives each protein 64 numbers. One thread is used, because that is the only way the result repeats.",
    "build_string_n2v": "Embeds the whole network, then picks the rows of the perturbed genes. A gene with no protein, or no strong link, is recorded as missing with the reason.",
    "string_ensembl_map": "Ensembl gene id to STRING protein id, for genes with exactly one protein.",
    "build_all_genes": "GenePT and STRING vectors for all 33,694 measured genes, with a table saying how each was matched. Nothing in Week 1 or 2 depends on it; it was built for later weeks.",
    "@BUILDERS": "Name of each method to the function that builds it.",
    "main": "Builds the requested embeddings.",
    "@main": "Runs `main`.",
}
N["src/pertact/decoders.py"] = {
    "@head": "The docstring gives both models as formulas. The embedding is the only input that changes between runs; `W` maps every embedding to the same width, so everything after it has the same capacity. `VAL_BATCH` is how many validation cells are scored at a time.",
    "pick_device": "CUDA if present, else Apple's MPS, else CPU.",
    "PertEmbedding": "A table from gene to vector. A perturbation's vector is the sum of its genes' rows, so a double is `e(a) + e(b)`. A gene the table does not cover adds nothing. `from_arrays` rescales the whole table by one number so that the average row length over the training genes is 1, the same rule for every embedding.",
    "SplitData": "One set of cells, ready for the model: the expression to predict, each cell's perturbation and gem group, and that set's control states. `sample_controls` picks, for each target cell, one random control from the same gem group.",
    "dense": "Sparse matrix to a plain array.",
    "build_split": "Builds a `SplitData` for training, validation or test. With `target_conds=[]` it loads controls only, which is how the test set is handled: its doubles are never loaded as training targets.",
    "mlp": "A stack of linear layers with layer norm, ReLU and dropout between them.",
    "LatentAdditive": "The model. The control state is standardised and encoded to 128 numbers. The embedding is mapped to 128 numbers by `W`, which has no bias, so an all-zero embedding means no perturbation. The two are added and decoded to all 5,028 genes. With `use_state=False` the control branch is dropped, which is the decoder-only model.",
    "@DECODERS": "The two decoder names, and whether each uses the control state.",
    "train_decoder": "Trains on training cells with squared error on expression. Every epoch pairs each target cell with a fresh random control from its gem group. Training stops when the loss on a fixed set of validation pairs has not improved for 8 epochs, and the best weights are restored.",
    "predict_means": "For one perturbation, draws 500 control states from the scored set, matched to the gem groups of that perturbation's real cells, runs them through the model and averages. This average is what gets scored.",
    "predict_means_compose": "For a double, compares two ways to combine its singles on the same controls: add the two embeddings and decode once, or apply one, re-encode the prediction with the frozen PCA, and apply the other (both orders, then averaged).",
}
N["src/pertact/metrics.py"] = {
    "@head": "The docstring defines every metric. All of them are computed per perturbation, then averaged over perturbations.",
    "group_means": "Average expression of each perturbation's cells within a set.",
    "split_mask": "True for the cells of one set.",
    "eval_conditions": "What is scored in each set: the 105 singles on validation, the 127 doubles on test.",
    "train_deltas": "For each single: its average change from control, measured on training cells. The additive baseline is built from these.",
    "compute_de": "Ranks the most changed genes for each perturbation with a Wilcoxon test and stores the top 50. || For singles this uses training cells. For doubles it uses the doubles' own test cells against test controls. That chooses which 20 genes a double is scored on; no model is fitted or tuned with it.",
    "load_de": "Reads the stored gene lists.",
    "pearson": "Pearson correlation of two vectors.",
    "EvalRef": "Everything that stays fixed while scoring one set: the measured average of each perturbation, the control average, and the 20 scored genes (with and without the perturbed genes themselves). For test it also holds the additive reference, control + change of a + change of b, and marks the quarter of doubles that depart from it most.",
    "summarise": "Averages the per-perturbation scores, overall, on the perturbations the embedding covers, and on the most non-additive doubles.",
    "spearman_brown_ceiling": "The formula behind the noise ceiling. From the correlation between the averages of two random halves of a group of cells, it gives the best correlation any model could reach against the whole group's average.",
    "noise_ceiling": "The best score any model could reach. Each double's cells are split in half and the two halves compared, which measures how noisy the measured average itself is.",
}
N["src/pertact/runs.py"] = {
    "@head": "The Week 1 experiment. Decoders are saved and reloaded by file name; predictions and scores go to `results/`.",
    "replace_runs": "Drops the rows of the runs being redone from a results table and appends the new ones.",
    "write_results": "Adds scores to `results/week1.csv`, replacing rows for runs that were just redone.",
    "fingerprint": "A short hash of an embedding table. It is saved with every decoder trained on that table, so a reloaded decoder can be matched to its embedding.",
    "run_baselines": "Scores the references that need no model: no change, the training average (validation), adding the two singles' changes (test), and the noise ceiling.",
    "pred_path": "File name for one run's saved predictions.",
    "model_path": "File name for one run's saved decoder.",
    "Runner": "Loads the data once and keeps it on the device. `train_or_load` reuses a saved decoder if its file exists, after checking its hyperparameters and the fingerprint of its embedding table; otherwise it trains and saves one. `fit_eval` predicts and scores the requested splits. A runner built with `eval_splits=(\"val\",)` never loads the test controls or test references. || Decoders saved before the fingerprint existed cannot be checked. They reload with a warning; delete the file in `models/decoders/` to retrain one.",
    "compose": "The sequential-composition test: scores the saved latent additive models when a double is applied one gene after the other, and checks that plain addition still reproduces the saved predictions.",
    "rescore": "Recomputes every metric from saved predictions, with no training. Used after a metric was added.",
    "tune": "Tries the decoder grid with the one-hot embedding and picks the best by the validation score. It can be stopped and resumed. The runner is built for validation only, so tuning cannot touch the test split.",
    "run": "The Week 1 experiment: every embedding, both decoders, three seeds, with the tuned hyperparameters.",
}
N["src/pertact/evaluate.py"] = {
    "@head": "The command line. Seven commands; each calls into `metrics.py`, `runs.py` or `decoder_test.py`.",
    "main": "Dispatches the seven commands.",
    "@main": "Runs `main`.",
}
N["src/pertact/ot.py"] = {
    "@head": "The docstring states the idea: no cell has a measured \"before\", so each perturbed cell is paired with the controls it most plausibly came from. The solver is the POT library's. `EPS_SUBSAMPLE` is how many controls are sampled to set the size of the entropy term.",
    "sinkhorn": "A thin wrapper around the POT library's log-domain Sinkhorn-Knopp solver, for two clouds whose cells all weigh the same. It finds the plan that moves the control cloud onto the perturbed cloud at the least total squared distance, softened by an entropy term of size `eps`. || Until 11 November the pairs came from a hand-written solver; what it produced is kept in `results/legacy_solver/`. POT gives the same plan up to small numerical differences; see the review list.",
    "pair": "One pairing problem. `eps` is 0.05 times the median squared distance between controls, so it does not grow when a perturbation moves cells far. After solving, each perturbed cell keeps its 5 heaviest controls, with weights rescaled to sum to 1. It also returns diagnostics, such as over how many controls a cell's weight is spread.",
    "run_space": "Runs every pairing problem in one state space, for training cells and for validation cells. First per perturbation and gem group. Then pooled: all single-perturbed cells of a gem group at once, without their labels, which the reverse test needs.",
    "load_pairs": "Reads one saved set of pairs.",
    "start_states": "The start state of every paired cell: the weighted average of its partner controls' states. The mean shift, both rescalings and the reverse test use it.",
    "check": "Two checks of the pairing on synthetic data: a known shift is recovered, and moving the controls leaves the pairs unchanged.",
    "main": "Runs the checks or the pairing.",
    "@main": "Runs `main`.",
}
N["src/pertact/actions.py"] = {
    "@head": "Everything about a set of action vectors that is neither training nor testing. Everything here uses training pairs only.",
    "actions_path": "File of one saved set of vectors.",
    "load_actions": "One saved set as a dictionary from gene to vector.",
    "save_actions": "Writes one set of action vectors, its training history and its settings.",
    "mean_shift_actions": "The reference. For each gene: the average, over its training pairs, of the perturbed cell's state minus the weighted state of its 5 partner controls. No training.",
    "calibrate_scale": "One number for a whole set of action vectors: the multiplier that brings control + vector closest to the partner, by least squares on training pairs. This is the 0.42 used for InfoNCE.",
    "per_action_scale": "The same least squares, gene by gene. The result is the projection of the mean shift onto the learned direction.",
    "length_ratio": "Length of each learned vector divided by the length of that gene's mean shift.",
    "effect_tiers": "Each gene's effect strength (the length of its training mean shift, in units of the typical distance between two training cells) and its third: weak, medium or strong. Used by the Week 2 tests and to label the doubles.",
    "length_table": "For each gene and each set: its length relative to the mean shift, and the cosine between the two.",
}
N["src/pertact/infonce.py"] = {
    "@head": "The docstring writes out the loss. A query is a control state plus the action vector; the keys are the batch's perturbed cells; the score is negative squared distance divided by a temperature.",
    "PairData": "Holds the OT pairs of one set. `batch` picks 32 perturbations and 8 pairs of each, and for every perturbed cell draws one of its 5 partner controls with probability equal to its OT weight.",
    "infonce_loss": "The loss itself. Each query must be closer to its own partner than to the other 255 cells, and each cell closer to its own query than to the other 255. With `rho` above zero it adds the length anchor: an extra pull of each query onto its own partner.",
    "train_actions": "Training. The action vectors start at zero, one row per gene, and are the only thing that is learned. Every 100 steps the loss on 20 fixed validation batches is checked, and training stops after 10 checks without improvement. If no check ever ran, it stops with a clear error.",
    "anchor": "The length-anchor experiment. For each anchor weight it trains on seed 0 and measures the length ratio on training pairs. It picks the smallest weight that brings the median ratio into the accepted range, then trains all three seeds. Validation scores are logged, not used for the choice.",
    "tune": "Trains once per temperature and state space on seed 0. || The temperature is chosen by the forward test's score on validation cells, so validation scores for InfoNCE are slightly optimistic.",
    "train_all": "Trains the final action vectors: both state spaces, three seeds, with the chosen temperature.",
    "main": "Dispatches `tune`, `train` and `anchor`.",
    "@main": "Runs `main`.",
}
N["src/pertact/retrieval.py"] = {
    "@head": "The docstring defines the two tests and lists every set of action vectors that is scored.",
    "_avg_rank": "Rank of the right answer among the candidates, with ties averaged.",
    "instance_retrieval": "The forward test. For each cell, the query is its top partner control plus the action vector, and all cells of the set are ranked by distance to it. Returns the mean reciprocal rank, recall at 1, 10 and 100, and how many of the 10 nearest cells have the right perturbation. || The query uses only the single heaviest partner (`partners[:, 0]`), while the reverse test below uses the weighted average of all five.",
    "identification": "The reverse test. The start state comes from the pooled pairs, or from the average control of the cell's gem group. All 105 action vectors plus \"no perturbation\" are tried, and the true perturbation is ranked by how close each lands to the cell.",
    "build_action_sets": "Builds every set of action vectors that gets tested in one state space: the mean shift, no action and shuffled first, then for each seed InfoNCE as trained, rescaled globally and per gene, with swapped lengths, and anchored. A seed with no saved vectors is reported and left out. || The shuffled set comes from a random permutation of the genes, which can leave a gene in its own place. With 105 genes that is about one gene on average.",
    "run_tests": "Splits the genes into weak, medium and strong thirds by the length of their mean shift, runs both tests on every set of vectors, on validation cells and on training cells, and writes the five result tables.",
    "main": "Runs the evaluation.",
    "@main": "Runs `main`.",
}
N["src/pertact/doubles.py"] = {
    "@head": "The docstring defines the two scores and says which lists were fixed before the first run on the doubles. `RULES` names the three ways a condition's vectors are combined: the sum, the stronger gene alone, and the better gene alone.",
    "real_shifts": "What really happened to each condition: the centre of its cells minus the centre of the controls of the same split, gem group by gem group, averaged with the condition's cell counts. Also `noise_sq`, the squared miss a perfect prediction would still have because both centres come from a finite number of cells.",
    "centre_scores": "Compares a predicted shift with the measured one: squared miss, the miss as a fraction of the squared real length (1 = no better than adding nothing), cosine for direction and length ratio for length.",
    "all_combinations": "Every candidate a cell could have received: no perturbation, each of the 105 genes alone, and each of the 5,460 pairs.",
    "compose": "Turns candidates into vectors: each row is the sum of its genes' action vectors, and zero for no gene.",
    "rank_true": "For every query, the rank of the right candidate when candidates are ordered by how close start + vector lands. The same formula as the Week 2 reverse test, in chunks so 5,566 candidates fit in memory.",
    "_rank_summary": "Top-1, top-5 and mean reciprocal rank.",
    "score_conditions": "The whole test for one state space, on plain arrays. For every action set it scores the sum, the stronger gene alone and the better gene alone against each condition's real shift, then names each condition from every cell and from its centre, for every candidate list. The last block counts in how many conditions a set beats the mean shift and its own single vectors. || The better-gene-alone reference looks at the double to choose the gene, so it is an upper bound for using one vector, not something that could be run blind.",
    "run": "Loads the data, builds the same action sets as the Week 2 tests, and scores them on the test doubles, or on validation singles with one gene per condition. || This is the only place in the package where the Week 2 vectors meet test cells. Nothing is fitted here, but the scores must not be used to choose a variant that is then reported on the same doubles.",
    "main": "Writes the two tables for the test doubles. With `--split val` it writes nothing and compares its naming scores with the saved Week 2 reverse test, which they must equal.",
    "@main": "Runs `main`.",
}
N["src/pertact/decoder_test.py"] = {
    "@head": "The short names of the two state spaces as used in file names, and the action-vector variants the test accepts.",
    "action_embedding": "Returns one Week 2 action-vector set as a gene-by-vector table: InfoNCE as trained, rescaled globally or per gene, anchored, or the mean shift. A `_unit` suffix divides every vector by its own length, keeping only its direction.",
    "action_smoke": "The decoder test. Each action-vector set is frozen and fed to both decoders. By default it is scored on validation cells only, and the runner is built without the test set. With `doubles=True` the same decoders also predict the 127 test doubles as the sum of the two genes' vectors, both sets of cells are scored, and the files go to `results/week3_decoder/`. || With `doubles=True` this reads test cells. The decoders are still trained on training cells and stopped on validation cells, so nothing is fitted to a double, but the list of variants to run must be fixed before looking at the scores.",
}
N["scripts/perturbation_map.py"] = {
    "@head": "Fixed example cells and genes for the Data tab's tables.",
    "main": "Collects the design (singles, doubles, how many cells each has), checks how much each guide raises its own gene on training cells, and saves a few example cells and condition averages. || The condition averages include two doubles, so this script reads test cells. They are only displayed.",
    "@main": "Runs `main`.",
}
N["scripts/report_tables.py"] = {
    "@head": "Display names for methods and metrics.",
    "summary": "Mean and standard deviation over seeds.",
    "cell": "One formatted table cell.",
    "@NEW_METRICS": "The metrics added after the first report.",
    "table": "One Markdown table for a set of perturbations.",
    "main": "Prints the report's tables, and optionally a comparison with an earlier results file.",
    "@main": "Runs `main`.",
}
N["scripts/week2_direction_diagnostics.py"] = {
    "@head": "Which action-vector sets are measured. The state spaces and seeds come from the config.",
    "main": "For every set of vectors as the decoder sees it: how similar the directions are to each other, how lengths differ between weak and strong perturbations, and how much of each weak direction lies in the space spanned by the strong ones.",
    "@main": "Runs `main`.",
}
N["scripts/week2_decoder_summary.py"] = {
    "@head": "Which sets are compared with fitted lengths against unit length.",
    "main": "Averages the decoder test's scores by seed, overall and by effect-strength third, and counts in how many seed-paired comparisons the weak third improves at unit length.",
    "@main": "Runs `main`.",
}
N["scripts/week3_week1_in_state_space.py"] = {
    "@head": "The docstring says why this works only in PCA-50 (it is a linear map of expression, scVI is not) and why the comparison is not like for like.",
    "score_shifts": "Scores one method's predicted shifts for the 127 doubles with the functions of `pertact.doubles`: miss, cosine and length for every double, and how often the right double is named from a cell and from its centre.",
    "main": "Projects the additive expression baseline and each saved Week 1 prediction of a double into PCA-50, subtracts the control centre that double is measured from, scores the result, and counts in how many doubles each run lands closer than the summed mean shifts. || It first checks the projection on test controls: their mean profile must land on the mean of their states. || STRING lacks vectors for 12 genes, so the summary also has a row over the 108 doubles it covers.",
    "@main": "Runs `main`.",
}
N["scripts/plot_ot_pairs.py"] = {
    "@head": "The docstring explains the two rows of the figure and how the 2-D slice is chosen. The colours are the notebook's chart palette.",
    "slice_axes": "Chooses the two plotted directions for one perturbation: along the mean shift, and the direction of largest remaining spread.",
    "panel": "Draws one panel: controls, perturbed cells, the pair lines and the mean-shift arrow.",
    "main": "Loads the saved pairs, checks that every partner is a training control of the same gem group, draws OT pairs and random pairs on shared axes, and saves the numbers next to the figure. || The default third perturbation, BCL2L11, is the one perturbation whose mean shift cannot be told from noise. It is shown as a \"weak\" example, which is misleading.",
    "@main": "Runs `main`.",
}
N["scripts/check_outputs.py"] = {
    "@head": "The steps: what each one runs, which saved files it needs as input, and which files it compares. Read-only inputs are linked; files a step rewrites are copied.",
    "saved": "The saved file that a regenerated file is compared with.",
    "flat": "Every leaf of a nested JSON value, with its path.",
    "num_diff": "Largest absolute difference between two numeric arrays. Two NaNs in the same place count as equal.",
    "compare": "Compares a regenerated file with the saved one: tables column by column, arrays key by key, JSON leaf by leaf. Returns the largest numeric difference and any other mismatch.",
    "run_step": "Runs one step with a temporary copy of the config whose output folders point into a temporary directory, then compares what the step wrote.",
    "main": "Runs the chosen steps and prints one line per compared file.",
    "@main": "Runs `main`.",
}
N["scripts/build_site.py"] = {
    "@head": "Site tooling.",
    "clean": "Replaces NaN with null, which JSON requires.",
    "read_json": "Reads a JSON file, or returns an empty object.",
    "inject": "Replaces the contents of one JSON block in the page.",
    "main": "Summarises `results/week1.csv` over seeds and writes it, with the tuning grid and the perturbation map, into the page.",
    "@main": "Runs `main`.",
}
N["scripts/build_walkthrough.py"] = {
    "@head": "Site tooling. The docstring says which numbers are typed by hand and which are read from files.",
    "fig": "Wraps an SVG figure with its caption.",
    "imgfig": "Wraps a PNG figure with its caption, and fails if the file is missing.",
    "ot_pair_rows": "Rows of the real-pairs table, read from the numbers saved by `plot_ot_pairs.py`.",
    "code": "Pulls a verbatim excerpt from a source file, between two marker strings.",
    "table": "A plain HTML table.",
    "trade": "A trade-off table with fixed column names.",
    "@code": "From here to `patch`, the Walkthrough's whole content is assembled as one long list of HTML strings: the prose, tables and figure calls for its twelve parts, then its CSS.",
    "patch": "Removes an earlier Walkthrough from the page and inserts the new one: styles, tab button, content and link handling.",
    "@code2": "Runs the build.",
    "mono": "Monospace text.",
    "nlines": "Counts the lines of files when the page is built, so the module table cannot go stale.",
    "@SCRIPTS": "The script and test files counted in the module table.",
}
N["scripts/walkthrough_figs.py"] = {
    "@head": "Site tooling. Colours used by the figures.",
    "Fig": "A small SVG helper: text, boxes, arrows, dots and tensor blocks, all on a 900-unit-wide canvas.",
    "fig_lines": "Figure: where the repository's lines are.",
    "fig_map": "Figure: which module feeds which.",
    "fig_idea": "Figure: a perturbation as an arrow, and the composition test.",
    "fig_funnel": "Figure: cell filtering, step by step.",
    "fig_split": "Figure: the three sets of cells.",
    "fig_tensors": "Figure: tensor shapes through preprocessing.",
    "fig_decoder": "Figure: the latent additive decoder with its tensors.",
    "fig_metric": "Figure: how one perturbation is scored.",
    "fig_w1": "Figure: the Week 1 result on the doubles.",
    "fig_ot": "Figure: one OT pairing problem.",
    "fig_infonce": "Figure: one InfoNCE step with its tensors.",
    "fig_tests": "Figure: the three tests of an action vector.",
    "@main": "Prints the size of every figure, as a quick check.",
}
N["scripts/build_code_tour.py"] = {
    "@head": "Site tooling. The docstring explains how the notes are kept in step with the code.",
    "read_lines": "A file as a list of lines.",
    "html_lines": "Colours the source with pygments, one HTML string per line.",
    "segments": "Cuts a file into functions, classes and the code between them, and checks no line is lost.",
    "ref": "Finds the current line number of a piece of code, for the review list.",
    "slug": "A file path as a link anchor.",
    "md": "Turns backticks in a note into code formatting.",
    "@GROUPS": "The files in run order with their roles, then the notes, one per function.",
    "fn": "A map box for a function or class.",
    "cmd": "A map box for a command.",
    "out": "A map box for a file the code writes.",
    "mod": "A map box for a whole file.",
    "txt": "A plain label on a map.",
    "@P": "The maps themselves: the pipeline, the notebook tooling, and one per main file. Each is a grid of boxes and a list of arrows.",
    "plain": "A note as plain text, safe to put in an HTML attribute.",
    "draw_map": "Draws one map as SVG. It checks that every label fits its box, that every function box names a real function, and that every arrow joins two boxes on the map.",
    "@MAP_JS": "The page script for the maps: highlight a box and its arrows on hover, describe it below the map, and step through the boxes with Previous and Next.",
    "review": "The review list at the top of the tab.",
    "render_file": "One file: its header, its map if it has one, then each segment with its note and source.",
    "build": "Assembles the whole tab.",
    "patch": "Removes an earlier Code tab from the page and inserts the new one.",
    "@CSS": "The tab's styles.",
    "@main": "Runs the build.",
}
N["scripts/wrap_preview.sh"] = {
    "@file": "Adds the doctype and character set the hosting page normally supplies, and rewrites image paths so the copy in `site/` finds the figures.",
}
N["pyproject.toml"] = {
    "@file": "The package name, where its code lives, the packages it imports at the versions the results were produced with, and the test settings.",
}


# ------------------------------------------------------------------ maps: clickable diagrams of the pipeline and of each main file
# Each map is a grid. A node sits at (column, row); an edge joins two node ids. Function nodes take their note from N
# and link to that function's source; command nodes link to the file's main(); file nodes are outputs and link nowhere.
def fn(name, col, row, sub, span=1, rel=None):
    """A box for a function or class. `rel` names the file it lives in when that is not the map's own file."""
    return dict(id=name, kind="fn", col=col, row=row, label=name, sub=sub, span=span, rel=rel)


def cmd(label, col, row, sub="", span=1, rel=None):
    return dict(id="cmd:" + label, kind="cmd", col=col, row=row, label=label, sub=sub, span=span, rel=rel)


def out(label, col, row, sub):
    return dict(id="file:" + label, kind="file", col=col, row=row, label=label, sub=sub, span=1)


def mod(path, col, row, sub, tag=""):
    return dict(id="mod:" + path + tag, kind="mod", col=col, row=row, label=Path(path).name, sub=sub, span=1, path=path)


def txt(label, col, row):
    return dict(id="txt:" + label, kind="txt", col=col, row=row, label=label, sub="", span=1)


P = "src/pertact/"
PIPELINE = dict(cols=6, title="The pipeline, file by file", nodes=[
    txt("Week 1", 0, 0), txt("Week 2", 0, 2), txt("Week 3", 0, 3),
    mod(P + "data.py", 0, 1, "1 · filter cells", "#1"), mod(P + "splits.py", 1, 1, "assign the sets"),
    mod(P + "data.py", 2, 1, "2 · choose genes", "#2"), mod(P + "encoders.py", 3, 1, "cell states"),
    mod(P + "embeddings.py", 3, 0, "gene embeddings"), mod(P + "decoders.py", 4, 0, "the test model"),
    mod(P + "runs.py", 5, 0, "train and score"),
    mod(P + "ot.py", 3, 2, "pair the cells"), mod(P + "infonce.py", 4, 2, "action vectors"),
    mod(P + "retrieval.py", 5, 2, "the two tests"), mod(P + "decoder_test.py", 5, 1, "vectors in decoder"),
    mod(P + "doubles.py", 5, 3, "sums on the doubles")],
    edges=[("mod:src/pertact/data.py#1", "mod:src/pertact/splits.py"), ("mod:src/pertact/splits.py", "mod:src/pertact/data.py#2"),
           ("mod:src/pertact/data.py#2", "mod:src/pertact/encoders.py"), ("mod:src/pertact/data.py#2", "mod:src/pertact/embeddings.py"),
           ("mod:src/pertact/embeddings.py", "mod:src/pertact/decoders.py"), ("mod:src/pertact/encoders.py", "mod:src/pertact/decoders.py"),
           ("mod:src/pertact/decoders.py", "mod:src/pertact/runs.py"), ("mod:src/pertact/encoders.py", "mod:src/pertact/ot.py"),
           ("mod:src/pertact/ot.py", "mod:src/pertact/infonce.py"), ("mod:src/pertact/infonce.py", "mod:src/pertact/retrieval.py"),
           ("mod:src/pertact/infonce.py", "mod:src/pertact/decoder_test.py"), ("mod:src/pertact/decoder_test.py", "mod:src/pertact/runs.py"),
           ("mod:src/pertact/infonce.py", "mod:src/pertact/doubles.py")])
TOOLING = dict(cols=5, title="How this notebook is built", nodes=[
    mod("scripts/walkthrough_figs.py", 0, 1, "draws the SVG figures"), mod("scripts/build_site.py", 1, 0, "Week 1 scores as JSON"),
    mod("scripts/build_walkthrough.py", 1, 1, "the Walkthrough tab"), mod("scripts/build_code_tour.py", 1, 2, "this tab"),
    out("site/index.html", 2, 1, "the published page"), mod("scripts/wrap_preview.sh", 3, 1, "adds doctype, fixes paths"),
    out("site/notebook.html", 4, 1, "opens from disk")],
    edges=[("mod:scripts/walkthrough_figs.py", "mod:scripts/build_walkthrough.py"), ("mod:scripts/build_site.py", "file:site/index.html"),
           ("mod:scripts/build_walkthrough.py", "file:site/index.html"), ("mod:scripts/build_code_tour.py", "file:site/index.html"),
           ("file:site/index.html", "mod:scripts/wrap_preview.sh"), ("mod:scripts/wrap_preview.sh", "file:site/notebook.html")])

MAPS = {}
MAPS[P + "data.py"] = dict(cols=4, nodes=[
    cmd("data qc", 0, 0, "stage 1"), fn("build_qc", 1, 0, "filter cells, keep all genes"), fn("annotate", 2, 0, "perturbation of each cell"),
    fn("parse_guide_id", 3, 0, "accept or reject a guide"), fn("construct_genes", 3, 1, "construct name to genes"),
    fn("cross_check_labels", 2, 1, "compare with scPerturb"), fn("resolve_targets_to_var", 2, 2, "gene to matrix column"),
    fn("SymbolResolver", 3, 2, "HGNC names"), fn("summarise_conditions", 2, 3, "cells per perturbation"),
    out("qc_allgenes.h5ad", 1, 1, "all genes, raw counts"),
    cmd("data hvg", 0, 4, "stage 2, after the split"), fn("build_processed", 1, 4, "gene choices on TR cells"),
    out("norman2019.h5ad", 2, 4, "5,028 genes, log-normalised")],
    edges=[("cmd:data qc", "build_qc"), ("build_qc", "annotate"), ("annotate", "parse_guide_id"), ("parse_guide_id", "construct_genes"),
           ("build_qc", "cross_check_labels"), ("build_qc", "resolve_targets_to_var"), ("resolve_targets_to_var", "SymbolResolver"),
           ("build_qc", "summarise_conditions"), ("build_qc", "file:qc_allgenes.h5ad"), ("file:qc_allgenes.h5ad", "build_processed"),
           ("cmd:data hvg", "build_processed"), ("build_processed", "file:norman2019.h5ad")])
MAPS[P + "splits.py"] = dict(cols=3, nodes=[
    cmd("splits", 0, 0, "run once, seed 0"), fn("make_splits", 1, 0, "20% of each single to VA, doubles to TE"),
    out("splits.json", 2, 0, "read by every later step")],
    edges=[("cmd:splits", "make_splits"), ("make_splits", "file:splits.json")])
MAPS[P + "encoders.py"] = dict(cols=4, nodes=[
    cmd("encoders pca", 0, 0), fn("fit_pca", 1, 0, "50 components, fitted on TR"), fn("save_state", 2, 0, "checks the cell order"),
    out("state_embeddings.npz", 3, 0, "X_pca_g, X_scvi10, X_scvi30"),
    cmd("encoders scvi", 0, 1), fn("fit_scvi", 1, 1, "trained on TR-cell counts"),
    fn("attach_state", 2, 1, "loads states, checks the order"), fn("load_all", 3, 1, "matrix + states + split"),
    cmd("encoders umap", 0, 2), fn("umap_figures", 1, 2, "three sanity pictures"),
    fn("scale_D0", 3, 2, "typical distance in a space")],
    edges=[("cmd:encoders pca", "fit_pca"), ("fit_pca", "save_state"), ("save_state", "file:state_embeddings.npz"),
           ("cmd:encoders scvi", "fit_scvi"), ("fit_scvi", "save_state"), ("cmd:encoders umap", "umap_figures"),
           ("file:state_embeddings.npz", "attach_state"), ("attach_state", "load_all")])
MAPS[P + "embeddings.py"] = dict(cols=5, nodes=[
    cmd("embeddings", 0, 0, "builds the four Week 1 embeddings, one after another", span=4), cmd("embeddings all_genes", 4, 0),
    fn("build_onehot", 0, 1, "identity, 105 genes"), fn("build_pca_loadings", 1, 1, "from TR pseudobulk"),
    fn("build_genept", 2, 1, "text-derived vectors"), fn("build_string_n2v", 3, 1, "network vectors"),
    fn("build_all_genes", 4, 1, "every measured gene"),
    fn("load_genept", 2, 2, "cached read"), fn("string_graph", 3, 2, "links with score ≥ 700"),
    fn("_stream_genept_pickle", 2, 3, "read without executing"), fn("node2vec_embed", 3, 3, "walks + skip-gram"),
    fn("save_embedding", 0, 4, "writes <method>.npz and updates coverage.json", span=4)],
    edges=[("cmd:embeddings", "build_onehot"), ("cmd:embeddings", "build_pca_loadings"), ("cmd:embeddings", "build_genept"),
           ("cmd:embeddings", "build_string_n2v"), ("cmd:embeddings all_genes", "build_all_genes"),
           ("build_genept", "load_genept"), ("load_genept", "_stream_genept_pickle"), ("build_string_n2v", "string_graph"),
           ("string_graph", "node2vec_embed"), ("build_onehot", "save_embedding"), ("build_pca_loadings", "save_embedding"),
           ("_stream_genept_pickle", "save_embedding"), ("node2vec_embed", "save_embedding")])
MAPS[P + "decoders.py"] = dict(cols=5, nodes=[
    fn("PertEmbedding", 0, 0, "gene to vector table"), fn("build_split", 0, 1, "one set of cells"),
    fn("SplitData", 1, 1, "targets + control pool"), fn("train_decoder", 2, 0, "MSE, early stop on VA"),
    fn("LatentAdditive", 3, 0, "enc(state) + W·e, decoded"), fn("mlp", 3, 1, "layer stack"),
    fn("predict_means", 4, 0, "500 controls, averaged"), fn("predict_means_compose", 4, 1, "added vs one by one")],
    edges=[("PertEmbedding", "train_decoder"), ("build_split", "SplitData"), ("SplitData", "train_decoder"),
           ("train_decoder", "LatentAdditive"), ("mlp", "LatentAdditive"), ("LatentAdditive", "predict_means"),
           ("LatentAdditive", "predict_means_compose")])
MAPS[P + "metrics.py"] = dict(cols=4, nodes=[
    fn("compute_de", 0, 0, "ranks the most changed genes"), out("de_genes.json", 1, 0, "50 ranked genes each"),
    fn("load_de", 2, 0, "reads the lists"), fn("EvalRef", 3, 0, "fixed references and metrics"),
    fn("train_deltas", 2, 1, "each single's mean change"), fn("summarise", 3, 1, "average over perturbations"),
    fn("pearson", 0, 2, "correlation"), fn("noise_ceiling", 1, 2, "best reachable score"),
    fn("spearman_brown_ceiling", 2, 2, "the ceiling formula")],
    edges=[("compute_de", "file:de_genes.json"), ("file:de_genes.json", "load_de"), ("load_de", "EvalRef"), ("train_deltas", "EvalRef"),
           ("EvalRef", "summarise"), ("pearson", "noise_ceiling"), ("noise_ceiling", "spearman_brown_ceiling")])
MAPS[P + "runs.py"] = dict(cols=4, nodes=[
    fn("tune", 0, 0, "decoder grid, chosen on VA"), fn("fingerprint", 2, 0, "hash of an embedding table"),
    fn("run", 0, 1, "every embedding, decoder, seed"), fn("Runner", 1, 1, "train or reload, predict, score"),
    fn("train_decoder", 2, 1, "trains one decoder", rel=P + "decoders.py"),
    fn("compose", 0, 2, "doubles, one gene at a time"), fn("predict_means", 2, 2, "mean prediction", rel=P + "decoders.py"),
    fn("rescore", 0, 3, "metrics from saved predictions"), fn("write_results", 1, 3, "adds scores to week1.csv"),
    fn("replace_runs", 2, 3, "drops the redone runs"), out("results/week1.csv", 3, 3, "one row per metric"),
    fn("run_baselines", 0, 4, "references that need no model")],
    edges=[("tune", "Runner"), ("run", "Runner"), ("compose", "Runner"), ("Runner", "fingerprint"), ("Runner", "train_decoder"),
           ("Runner", "predict_means"), ("run", "write_results"), ("rescore", "write_results"), ("run_baselines", "write_results"),
           ("write_results", "replace_runs"), ("replace_runs", "file:results/week1.csv")])
MAPS[P + "evaluate.py"] = dict(cols=4, nodes=[
    cmd("evaluate de", 0, 0), fn("compute_de", 1, 0, "ranks the most changed genes", rel=P + "metrics.py"), out("de_genes.json", 2, 0, "50 ranked genes each"),
    cmd("evaluate baselines", 0, 1), fn("run_baselines", 1, 1, "references that need no model", rel=P + "runs.py"),
    fn("noise_ceiling", 2, 1, "best reachable score", rel=P + "metrics.py"),
    cmd("evaluate tune", 0, 2), fn("tune", 1, 2, "decoder grid, chosen on VA", rel=P + "runs.py"),
    cmd("evaluate run", 0, 3), fn("run", 1, 3, "every embedding, decoder, seed", rel=P + "runs.py"),
    fn("Runner", 2, 3, "train or reload, predict, score", rel=P + "runs.py"), fn("EvalRef", 3, 3, "fixed references and metrics", rel=P + "metrics.py"),
    cmd("evaluate compose", 0, 4), fn("compose", 1, 4, "doubles, one gene at a time", rel=P + "runs.py"), out("results/week1.csv", 3, 4, "one row per metric"),
    cmd("evaluate actions", 0, 5), fn("action_smoke", 1, 5, "decoder test, VA or doubles", rel=P + "decoder_test.py"),
    fn("action_embedding", 2, 5, "one set of action vectors", rel=P + "decoder_test.py"),
    cmd("evaluate rescore", 0, 6), fn("rescore", 1, 6, "metrics from saved predictions", rel=P + "runs.py")],
    edges=[("cmd:evaluate de", "compute_de"), ("compute_de", "file:de_genes.json"), ("cmd:evaluate baselines", "run_baselines"),
           ("run_baselines", "noise_ceiling"), ("cmd:evaluate tune", "tune"), ("tune", "Runner"), ("cmd:evaluate run", "run"),
           ("run", "Runner"), ("Runner", "EvalRef"), ("EvalRef", "file:results/week1.csv"), ("cmd:evaluate compose", "compose"),
           ("compose", "Runner"), ("cmd:evaluate actions", "action_smoke"), ("action_smoke", "action_embedding"),
           ("action_smoke", "Runner"), ("cmd:evaluate rescore", "rescore")])
MAPS[P + "ot.py"] = dict(cols=4, nodes=[
    cmd("ot --check", 0, 0), fn("check", 1, 0, "a known shift is recovered"),
    cmd("ot", 0, 1), fn("run_space", 1, 1, "every problem, both pairings"), fn("pair", 2, 1, "set ε, solve, keep 5 controls"),
    fn("sinkhorn", 3, 1, "the POT library's solver"),
    out("results/ot/*.npz", 2, 2, "partners and weights per cell"), fn("load_pairs", 3, 2, "used by actions, infonce, tests"),
    fn("start_states", 3, 3, "weighted average of partners")],
    edges=[("cmd:ot --check", "check"), ("check", "pair"), ("cmd:ot", "run_space"), ("run_space", "pair"),
           ("pair", "sinkhorn"), ("run_space", "file:results/ot/*.npz"), ("file:results/ot/*.npz", "load_pairs"), ("load_pairs", "start_states")])
MAPS[P + "actions.py"] = dict(cols=3, nodes=[
    fn("load_pairs", 0, 0, "the saved OT pairs of training cells", rel=P + "ot.py"), fn("mean_shift_actions", 1, 0, "the reference, no training"),
    fn("length_ratio", 2, 0, "length against the mean shift"),
    fn("calibrate_scale", 1, 1, "one number for a whole set"), fn("length_table", 2, 1, "length and cosine, per gene"),
    fn("per_action_scale", 1, 2, "one number per gene"),
    fn("save_actions", 0, 3, "vectors, history and settings"), out("infonce__<space>__s<seed>.npz", 1, 3, "one vector per gene"),
    fn("load_actions", 2, 3, "gene to vector")],
    edges=[("load_pairs", "mean_shift_actions"), ("load_pairs", "calibrate_scale"), ("load_pairs", "per_action_scale"),
           ("mean_shift_actions", "length_ratio"), ("mean_shift_actions", "length_table"),
           ("save_actions", "file:infonce__<space>__s<seed>.npz"), ("file:infonce__<space>__s<seed>.npz", "load_actions")])
MAPS[P + "infonce.py"] = dict(cols=4, nodes=[
    cmd("infonce tune", 0, 0), fn("tune", 1, 0, "temperature, by forward test"), fn("PairData", 3, 0, "batches of OT pairs"),
    cmd("infonce train", 0, 1), fn("train_all", 1, 1, "2 state spaces × 3 seeds"), fn("train_actions", 2, 1, "the training loop"),
    fn("infonce_loss", 3, 1, "the loss"),
    cmd("infonce anchor", 0, 2), fn("anchor", 1, 2, "anchor weight, by length ratio"),
    fn("save_actions", 2, 2, "vectors and history", rel=P + "actions.py"), fn("scale_D0", 3, 2, "the distance unit", rel=P + "encoders.py"),
    fn("length_ratio", 2, 3, "length against the mean shift", rel=P + "actions.py"), out("infonce__*.npz", 3, 3, "one vector per gene")],
    edges=[("cmd:infonce tune", "tune"), ("tune", "train_actions"), ("cmd:infonce train", "train_all"), ("train_all", "train_actions"),
           ("train_all", "save_actions"), ("cmd:infonce anchor", "anchor"), ("anchor", "train_actions"), ("anchor", "save_actions"),
           ("anchor", "length_ratio"), ("train_actions", "PairData"), ("train_actions", "infonce_loss"), ("train_actions", "scale_D0"),
           ("save_actions", "file:infonce__*.npz")])
MAPS[P + "retrieval.py"] = dict(cols=4, nodes=[
    fn("build_action_sets", 2, 0, "every set that is tested"), fn("mean_shift_actions", 3, 0, "and the two rescalings", rel=P + "actions.py"),
    cmd("retrieval", 0, 1), fn("run_tests", 1, 1, "every vector set, both tests"),
    fn("instance_retrieval", 2, 1, "forward test: find the cell"), fn("_avg_rank", 3, 1, "rank of the right answer"),
    fn("identification", 2, 2, "reverse test: name the gene"),
    fn("length_table", 2, 3, "length and cosine vs mean shift", rel=P + "actions.py"), out("results/week2_*.csv", 3, 3, "five result tables")],
    edges=[("cmd:retrieval", "run_tests"), ("run_tests", "build_action_sets"), ("build_action_sets", "mean_shift_actions"),
           ("run_tests", "instance_retrieval"), ("run_tests", "identification"), ("run_tests", "length_table"),
           ("instance_retrieval", "_avg_rank"), ("identification", "_avg_rank"), ("length_table", "file:results/week2_*.csv")])
MAPS[P + "doubles.py"] = dict(cols=4, nodes=[
    cmd("doubles", 0, 1), fn("run", 1, 1, "both state spaces"),
    fn("build_action_sets", 2, 0, "the same sets as Week 2", rel=P + "retrieval.py"), fn("effect_tiers", 3, 0, "weak, medium, strong", rel=P + "actions.py"),
    fn("score_conditions", 2, 1, "every set, both scores"), fn("real_shifts", 3, 1, "what really happened"),
    fn("all_combinations", 1, 2, "the 5,566 candidates"), fn("compose", 2, 2, "sum the genes' vectors"), fn("centre_scores", 3, 2, "miss, cosine, length"),
    fn("rank_true", 2, 3, "name the double"), fn("_avg_rank", 3, 3, "rank of the right answer", rel=P + "retrieval.py"),
    out("results/week3_doubles*.csv", 1, 3, "per set and per double")],
    edges=[("cmd:doubles", "run"), ("run", "build_action_sets"), ("run", "effect_tiers"), ("run", "score_conditions"), ("run", "all_combinations"),
           ("score_conditions", "real_shifts"), ("score_conditions", "centre_scores"), ("score_conditions", "compose"),
           ("score_conditions", "rank_true"), ("rank_true", "_avg_rank"), ("run", "file:results/week3_doubles*.csv")])
MAPS[P + "decoder_test.py"] = dict(cols=4, nodes=[
    cmd("evaluate actions", 0, 0, rel=P + "evaluate.py"), fn("action_smoke", 1, 0, "every variant, both decoders"),
    fn("action_embedding", 2, 0, "one set of action vectors"), fn("load_actions", 3, 0, "the saved vectors", rel=P + "actions.py"),
    fn("Runner", 2, 1, "validation, or with doubles", rel=P + "runs.py"), out("week2_decoder/smoke__*.csv", 3, 1, "scores of every run")],
    edges=[("cmd:evaluate actions", "action_smoke"), ("action_smoke", "action_embedding"), ("action_embedding", "load_actions"),
           ("action_smoke", "Runner"), ("Runner", "file:week2_decoder/smoke__*.csv")])
ROLE = {f["path"]: f["role"] for _, _, fs in GROUPS for f in fs}


def plain(s):
    return html.escape(s.replace("`", ""), quote=True)


def draw_map(mid, spec, rel=None, title=None):
    """One clickable SVG map plus its Previous / Next bar and the line that describes the selected box."""
    W, pad, nh, rp = 940, 12, 44, 66
    cols, nodes = spec["cols"], spec["nodes"]
    gap = 22 if cols >= 7 else 28 if cols == 6 else 24 if cols == 5 else 40
    nw = (W - 2 * pad - (cols - 1) * gap) / cols
    H = 2 * pad + (max(n["row"] for n in nodes)) * rp + nh
    seg_keys = {}

    def keys_of(r):
        if r not in seg_keys:
            seg_keys[r] = {k for k, *_ in segments(r, read_lines(r))}
        return seg_keys[r]

    def box(n):
        return pad + n["col"] * (nw + gap), pad + n["row"] * rp, nw * n["span"] + gap * (n["span"] - 1), nh

    by, svg = {n["id"]: n for n in nodes}, []
    for a, b in spec["edges"]:
        assert a in by and b in by, f"{mid}: edge {a} -> {b} names a node that is not in the map"
        (ax, ay, aw, ah), (bx, by_, bw, bh) = box(by[a]), box(by[b])
        overlap = min(ax + aw, bx + bw) - max(ax, bx)
        if overlap > 20 and ay != by_:  # stacked: straight down (or up) the shared column
            x = bx + bw / 2 if bw <= aw else ax + aw / 2
            y1, y2 = (ay + ah, by_ - 5) if by_ > ay else (ay, by_ + bh + 5)
            d = f"M{x:.1f},{y1:.1f} L{x:.1f},{y2:.1f}"
        else:  # side by side: leave the facing sides, bend smoothly if the rows differ
            fwd = bx > ax
            x1, x2 = (ax + aw, bx - 5) if fwd else (ax, bx + bw + 5)
            y1, y2, k = ay + ah / 2, by_ + bh / 2, (gap * 0.6 if fwd else -gap * 0.6)
            d = f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" if y1 == y2 else f"M{x1:.1f},{y1:.1f} C{x1 + k:.1f},{y1:.1f} {x2 - k:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
        svg.append(f'<path class="cm-e" d="{d}" data-a="{plain(a)}" data-b="{plain(b)}" marker-end="url(#{mid}-ar)"/>')
    for n in nodes:
        x, y, w, h = box(n)
        kind, label, sub = n["kind"], n["label"], n["sub"]
        if kind == "txt":
            svg.append(f'<text class="cm-lane" x="{x:.1f}" y="{y + h / 2 + 4:.1f}">{html.escape(label)}</text>')
            continue
        assert len(label) * 7.0 <= w - 14, f"{mid}: label too long for its box: {label}"
        assert len(sub) * 5.9 <= w - 14, f"{mid}: sub-label too long for its box: {sub}"
        flag, href, home = "", None, n.get("rel") or rel
        if kind == "fn":
            assert label in keys_of(home), f"{mid}: {label} is not a function or class of {home}"
            note, _, flag = (s.strip() for s in N[home][label].partition("||"))
            name, href = (label if home == rel else f"{Path(home).name}: {label}"), f"#{slug(home)}--{label}"
            kind = "fn" if home == rel else "fn ext"
        elif kind == "cmd":
            name = "python -I -m pertact." + label
            note, href = "Command. " + (sub[:1].upper() + sub[1:] + "." if sub else "It runs the functions it points to."), f"#{slug(home)}--main"
        elif kind == "mod":
            name, note, href = n["path"], ROLE[n["path"]], f"#{slug(n['path'])}"
        else:
            name, note = label, "A file this code writes. " + sub[:1].upper() + sub[1:] + "."
        tag, link = ("a", f' href="{href}"') if href else ("g", ' tabindex="0"')
        svg.append(
            f'<{tag} class="cm-n {kind}"{link} data-id="{plain(n["id"])}" data-name="{plain(name)}" data-note="{plain(note)}" data-flag="{plain(flag)}">'
            f'<title>{plain(note)}</title><rect class="cm-b" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h}" rx="{22 if kind == "cmd" else 6}"/>'
            f'<text class="cm-l" x="{x + (16 if kind == "cmd" else 10):.1f}" y="{y + (18 if sub else 27):.1f}">{html.escape(label)}</text>'
            + (f'<text class="cm-s" x="{x + (16 if kind == "cmd" else 10):.1f}" y="{y + 34:.1f}">{html.escape(sub)}</text>' if sub else "")
            + (f'<circle class="cm-dot" cx="{x + w - 11:.1f}" cy="{y + 11:.1f}" r="4"/>' if flag else "") + f"</{tag}>")
    has_flag = any(n["kind"] == "fn" and "||" in N[n.get("rel") or rel][n["label"]] for n in nodes)
    has_ext = any(n["kind"] == "fn" and n.get("rel") not in (None, rel) for n in nodes)
    return (f'<div class="cm" id="{mid}">\n<div class="cm-bar"><b>{html.escape(title or spec.get("title", "Map of this file"))}</b>'
            f'<button type="button" class="cm-prev" id="{mid}-prev">Previous</button><button type="button" class="cm-next" id="{mid}-next">Next</button>'
            f'<span class="cm-count">Step through in run order</span>'
            + ('<span class="cm-legend">' + ("<i></i>has a review note" if has_flag else "") + (" · " if has_flag and has_ext else "")
               + ("dashed: lives in another file" if has_ext else "") + "</span>" if has_flag or has_ext else "") + "</div>\n"
            f'<div class="cm-wrap"><svg viewBox="0 0 {W} {H}" role="group" aria-label="{html.escape(title or spec.get("title", "Map of " + (rel or "")))}" style="min-width:{680 if cols < 6 else 760}px">'
            f'<defs><marker id="{mid}-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path class="cm-ah" d="M0,0 L10,5 L0,10 z"/></marker></defs>\n'
            + "\n".join(svg) + "\n</svg></div>\n"
            '<p class="cm-info">Point at a box to see what it does. Click it to open its code. Or use Previous and Next to walk through in run order.</p>\n</div>')


MAP_JS = r'''<script>
(function () {
  "use strict";
  try {
    document.querySelectorAll(".cm").forEach(function (cm) {
      var svg = cm.querySelector("svg"), info = cm.querySelector(".cm-info"), count = cm.querySelector(".cm-count");
      var nodes = [].slice.call(cm.querySelectorAll("[data-id]")), edges = [].slice.call(cm.querySelectorAll("[data-a]"));
      var rest = info.textContent, cur = -1;
      function find(id) { for (var i = 0; i < nodes.length; i++) if (nodes[i].getAttribute("data-id") === id) return nodes[i]; return null; }
      function show(n) {
        nodes.forEach(function (x) { x.classList.remove("cm-on", "cm-near"); });
        edges.forEach(function (e) { e.classList.remove("cm-on"); });
        if (!n) { svg.classList.remove("cm-dim"); info.textContent = rest; return; }
        svg.classList.add("cm-dim");
        n.classList.add("cm-on");
        var id = n.getAttribute("data-id");
        edges.forEach(function (e) {
          var a = e.getAttribute("data-a"), b = e.getAttribute("data-b");
          if (a !== id && b !== id) return;
          e.classList.add("cm-on");
          var o = find(a === id ? b : a);
          if (o) o.classList.add("cm-near");
        });
        info.textContent = "";
        var name = document.createElement("b");
        name.textContent = n.getAttribute("data-name");
        info.appendChild(name);
        info.appendChild(document.createTextNode(" " + n.getAttribute("data-note")));
        var flag = n.getAttribute("data-flag");
        if (flag) { var f = document.createElement("span"); f.className = "cm-flagtxt"; f.textContent = " Review: " + flag; info.appendChild(f); }
        var href = n.getAttribute("href");
        if (href) { var a2 = document.createElement("a"); a2.href = href; a2.className = "cm-open"; a2.textContent = "Open its code"; info.appendChild(document.createTextNode(" ")); info.appendChild(a2); }
      }
      function held() { return cur >= 0 ? nodes[cur] : null; }
      function step(d) {
        cur = cur < 0 ? (d > 0 ? 0 : nodes.length - 1) : (cur + d + nodes.length) % nodes.length;
        show(nodes[cur]);
        count.textContent = "Step " + (cur + 1) + " of " + nodes.length;
        try { nodes[cur].scrollIntoView({ block: "nearest", inline: "center" }); } catch (e) {}
      }
      nodes.forEach(function (n) {
        n.addEventListener("pointerenter", function (e) { if (e.pointerType === "mouse") show(n); });
        n.addEventListener("pointerleave", function (e) { if (e.pointerType === "mouse") show(held()); });
        n.addEventListener("focus", function () { show(n); });
        n.addEventListener("blur", function () { show(held()); });
      });
      cm.querySelector(".cm-prev").addEventListener("click", function () { step(-1); });
      cm.querySelector(".cm-next").addEventListener("click", function () { step(1); });
    });
  } catch (e) {}
})();
</script>
'''


# ------------------------------------------------------------------ review list
def review():
    r = ref
    # needles with quotes in them, kept out of the f-strings below
    q = {"conds": 'default=["CEBPA"', "de": '("doubles", "test"', "fwd": 'Q = S[z["partners"][:, 0]] + F', "mrr": 'sort_values("mrr"',
         "fp": 'ck["embedding"] != fingerprint(emb)'}
    groups = [
        ("Behaviour a reader should know about", [
            f"<b>The scored genes of a double come from its own test cells.</b> See {r('src/pertact/metrics.py', q['de'])}. This follows the original brief. It chooses what is scored; it fits and tunes nothing.",
            f"<b>The forward and reverse tests build their start state differently.</b> The forward test uses the single heaviest partner ({r('src/pertact/retrieval.py', q['fwd'])}); the reverse test uses the weighted average of five ({r('src/pertact/retrieval.py', 's0 = start_states(S, z)')}). With the averaged start the forward test scores higher for every set of vectors.",
            f"<b>The InfoNCE temperature is chosen on validation cells.</b> See {r('src/pertact/infonce.py', q['mrr'])}. Validation scores for InfoNCE are therefore slightly optimistic. The length-anchor weight is chosen on training pairs only.",
            f"<b>The shuffled reference can leave a gene in place.</b> The permutation at {r('src/pertact/retrieval.py', 'perm = rng.permutation')} is not forced to move every gene. About one of 105 keeps its own vector, which lifts that reference very slightly.",
            f"<b>The number of partners and the size of the entropy term were never tuned.</b> {r('config.yaml', 'eps_scale: 0.05')} and the line after it are starting values. The 5 heaviest controls hold a median 23% of a cell's OT weight.",
            f"<b>The Week 2 results were regenerated with the POT solver on 11 November.</b> The code calls the POT library ({r('src/pertact/ot.py', 'pot.sinkhorn(')}). The hand-written solver used before gave a different set of partners for at most 0.16% of the cells in a file, and validation scores on the two sets of pairs differ by at most 0.005 (median 0.00015). The earlier solver's pairs, vectors and scores are kept in <code>results/legacy_solver/</code>, with tables of the differences.",
            "<b>The decoder test was not rerun on the new vectors.</b> The scores in <code>results/week2_decoder/</code> come from decoders trained on vectors from the earlier solver's pairs. Those 60 decoders were moved to <code>models/decoders_legacy_solver/</code>, so <code>evaluate actions</code> trains new ones instead of reloading them. The new vectors are nearly the same: median cosine 0.9998 for InfoNCE, above 0.999 for every mean-shift vector.",
            f"<b>Decoders saved before 11 November carry no embedding fingerprint.</b> The check at {r('src/pertact/runs.py', q['fp'])} applies to decoders saved from now on. Older files reload with a warning.",
            f"<b>Two places read test cells for display.</b> The UMAP figures ({r('src/pertact/encoders.py', 'def umap_figures')}) and the Data tab's condition averages ({r('scripts/perturbation_map.py', 'MEAN_CONDITIONS =')}). Neither feeds a model or a choice.",
        ]),
        ("Corrections still to make in the write-ups", [
            f"<b>The pair figure's weak example is the one perturbation with no detectable shift.</b> BCL2L11 is a default at {r('scripts/plot_ot_pairs.py', q['conds'])}. The Walkthrough presents its shift of 1.8 as real; on all 363 training cells it is 0.14 against a noise level of 0.12.",
            "<b>One sentence in <code>REPORT_week2.md</code> is misleading.</b> It says the weak third is \"near chance\" in the reverse test and links that to target genes that barely rise. The scores are 6 to 9 times chance, and per-cell noise explains them better.",
        ]),
        ("Safeguards in the code", [
            f"Every cell is in exactly one split, and no double-perturbation cell is in training or validation: {r('src/pertact/splits.py', 'split is not a partition of the cells')} and the line after it.",
            f"State vectors are checked against the cell order every time they are saved or loaded: {r('src/pertact/encoders.py', 'def attach_state')}.",
            f"Tuning and the Week 2 decoder test build a runner that never loads the test controls or test references: {r('src/pertact/runs.py', 'val-only runners never read test cells')}.",
            f"A decoder is matched to its embedding table by a fingerprint before it is reused: {r('src/pertact/runs.py', 'def fingerprint')}.",
            f"The downloaded GenePT pickle is read without being executed: {r('src/pertact/embeddings.py', 'def _stream_genept_pickle')}.",
            f"The saved results can be regenerated and compared: {r('scripts/check_outputs.py', 'STEPS = {')}. Fifteen tests cover the building blocks on synthetic inputs.",
        ]),
    ]
    out = []
    for title, items in groups:
        out.append(f'<h4 class="ct-rh">{title}</h4>\n<ul class="clean">' + "".join(f"<li>{i}</li>" for i in items if i) + "</ul>")
    return "\n".join(out)


# ------------------------------------------------------------------ rendering
def render_file(f, missing):
    rel = f["path"]
    lines = read_lines(rel)
    while lines and not lines[-1].strip():
        lines.pop()
    hl = html_lines(rel, lines)
    notes = dict(N.get(rel, {}))
    segs = segments(rel, lines)
    unknown = set(notes) - {k for k, *_ in segs}
    assert not unknown, f"{rel}: notes for segments that no longer exist: {sorted(unknown)}"
    meta = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in (
        ("Run", "<br>".join(f"<code>{html.escape(c)}</code>" for c in f.get("run", []))), ("Reads", md(f.get("reads", ""))),
        ("Writes", md(f.get("writes", ""))), ("Cells it reads", md(f["cells"]))) if v)
    out = [f'<article class="ct-file" id="{slug(rel)}">\n<header><div class="ct-path">{rel}</div>'
           f'<p>{md(f["role"])}</p><dl class="ct-meta">{meta}</dl></header>']
    has_map = rel in MAPS
    if has_map:
        out.append(draw_map(f"{slug(rel)}--map", MAPS[rel], rel))
    up = f' <a class="ct-up" href="#{slug(rel)}--map">back to the map</a>' if has_map else ""
    for key, label, a, b in segs:
        raw = notes.get(key)
        if raw is None and rel.startswith("tests/"):  # a test's name says what it checks
            raw = (key[5:].replace("_", " ").capitalize() + "." if key.startswith("test_") else
                   "Imports. The docstring says what the file covers." if key == "@head" else "A small input shared by the tests below.")
        if raw is None:
            missing.append(f"{rel}: {label}")
            raw = ""
        note, _, flag = (s.strip() for s in raw.partition("||"))
        body = "\n".join(f'<span class="l"><i data-n="{n}"></i>{hl[n - 1]}</span>' for n in range(a, b + 1))
        out.append(
            f'<div class="ct-seg" id="{slug(rel)}--{re.sub(r"[^A-Za-z0-9_]", "", key)}">'
            f'<h4>{html.escape(label)} <small>lines {a}–{b}</small>{up}</h4>'
            + (f"<p>{md(note)}</p>" if note else "")
            + (f'<p class="ct-flag"><span class="chip warn">Review</span> {md(flag)}</p>' if flag else "")
            + f'<details class="ct-src"{"" if f.get("closed") else " open"}><summary>Source, {b - a + 1} lines</summary>'
            f'<pre>{body}</pre></details></div>')
    out.append("</article>")
    return "\n".join(out), len(lines), len(segs)


def build():
    missing, body, rows, total, nseg = [], [], [], 0, 0
    for gi, (title, lede, files) in enumerate(GROUPS, 1):
        body.append(f'<div class="block ct-group" id="ct-g{gi}"><div class="ct-kicker">Part {gi} of {len(GROUPS)}</div>'
                    f'<h2>{title}</h2><p class="lede">{lede}</p>'
                    + (draw_map("ct-tooling", TOOLING) if gi == len(GROUPS) else "") + "</div>")
        for f in files:
            h, n, s = render_file(f, missing)
            body.append(h)
            total += n
            nseg += s
            rows.append(f'<tr><td><a href="#{slug(f["path"])}"><code>{f["path"]}</code></a></td><td class="num">{n:,}</td>'
                        f'<td>{md(f["role"])}</td><td>{md(f["cells"])}</td></tr>')
    listed = {f["path"] for _, _, fs in GROUPS for f in fs}
    on_disk = {str(p.relative_to(ROOT)) for d in ("src/pertact", "scripts", "tests") for p in (ROOT / d).glob("*") if p.suffix in (".py", ".sh")}
    assert on_disk <= listed, f"source files missing from the Code tab: {sorted(on_disk - listed)}"
    for m in missing:
        print("no note yet:", m)
    top = f'''<section class="week" id="codetour" aria-label="Code">
<div class="wrap ct">
  <div class="block" id="ct-top">
    <div class="ct-kicker">Code · full source with notes</div>
    <h2>Every file, in the order it runs</h2>
    <p class="lede">This tab shows all {total:,} lines of code and settings in the repository: {len(listed)} files, cut into {nseg} functions and blocks. Each block has a note saying what it does, then its complete source with the real line numbers.</p>
    <ul class="clean">
      <li><b>Read it in order.</b> The seven parts follow the pipeline: foundations, Week 1, Week 2, Week 3, analysis scripts, checks and tests, then the tooling for this notebook.</li>
      <li><b>"Cells it reads"</b> says which cells a file touches: TR is training cells, VA is validation cells, TE is test cells (the double perturbations).</li>
      <li><b>Maps are clickable.</b> Point at a box to see what it does and what it connects to, click it to open its code, or press Next to walk through a file in run order. An orange dot marks a box with a review note, and a dashed box is a function that lives in another file.</li>
      <li><b>A "Review" tag</b> marks something I would raise in a code review. All of them are collected in the list below.</li>
      <li><b>The listings are read from the files when the page is built,</b> so they cannot differ from the code. The notes are written by hand; the build stops if a note refers to a function that no longer exists.</li>
    </ul>
  </div>
  <div class="block" id="ct-map">
    <h3>Start from the map</h3>
    <p class="lede">Each box is a file. Click one to go to its section. {len(MAPS)} files open with their own map, where each box is a function; clicking it opens that function's source, and "back to the map" brings you back. Three files are not on this map: <code>metrics.py</code> and <code>actions.py</code> are libraries the boxes call, and <code>evaluate.py</code> is the command line.</p>
    {draw_map("ct-pipeline", PIPELINE)}
  </div>
  <div class="block" id="ct-files">
    <h3>The files</h3>
    <div class="tablewrap"><table><thead><tr><th>File</th><th class="num">Lines</th><th>What it does</th><th>Cells it reads</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div>
  </div>
  <div class="block callout" id="ct-review">
    <h3>Review notes</h3>
    <p class="lede">Open points from reading every line, as of {REVIEWED}, after the refactor. The file links are computed from the current source, so they point at the right line.</p>
    {review()}
  </div>
'''
    return top + "\n".join(body) + "\n</div>\n" + MAP_JS + "</section>\n", total, len(listed), nseg


CSS = '''/* Code tab */
#w-tour:checked ~ main #codetour { display: block; }
#w-tour:checked ~ nav.weeks label[for="w-tour"] { color: var(--ink); border-bottom-color: var(--ink); }
#w-tour:focus-visible ~ nav.weeks label[for="w-tour"] { outline: 2px solid var(--accent); outline-offset: 2px; }
.ct { padding-block: 28px 0; display: grid; gap: 36px; }
.ct > * { min-width: 0; }
.ct-kicker { font: 500 var(--step--1)/1.4 var(--f-mono); letter-spacing: .06em; text-transform: uppercase; color: var(--accent); }
.ct-group { padding-top: 20px; border-top: 2px solid var(--ink); gap: 6px; scroll-margin-top: 12px; }
.ct-rh { margin: 10px 0 0; font: 600 var(--step-0)/1.4 var(--f-body); }
#ct-review ul.clean { max-width: 86ch; }
#ct-files td:nth-child(3) { min-width: 260px; } #ct-files td:nth-child(4) { min-width: 200px; }
.ct-file { display: grid; gap: 22px; min-width: 0; scroll-margin-top: 12px; }
.ct-file > header { display: grid; gap: 8px; }
.ct-path { font: 500 var(--step-1)/1.3 var(--f-mono); overflow-wrap: anywhere; }
.ct-meta { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 4px 14px; margin: 0; font-size: var(--step--1); }
.ct-meta dt { color: var(--muted); } .ct-meta dd { margin: 0; overflow-wrap: anywhere; }
.ct-seg { display: grid; gap: 8px; min-width: 0; scroll-margin-top: 12px; }
.ct-seg h4 { margin: 0; font: 500 var(--step-0)/1.4 var(--f-mono); display: flex; flex-wrap: wrap; gap: 2px 10px; align-items: baseline; }
.ct-seg h4 small { font: 400 var(--step--1)/1.4 var(--f-body); color: var(--muted); }
.ct-seg > p { max-width: 82ch; }
.ct-flag .chip { margin-right: 6px; vertical-align: 1px; }
details.ct-src { background: var(--card); border: 1px solid var(--rule); border-radius: 8px; min-width: 0; }
details.ct-src summary { cursor: pointer; padding: 8px 14px; font-size: var(--step--1); color: var(--muted); }
details.ct-src summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
details.ct-src[open] summary { border-bottom: 1px solid var(--rule); }
details.ct-src pre { border-radius: 0 0 8px 8px; padding-left: 0; tab-size: 4; }
details.ct-src pre i { display: inline-block; width: 3.6em; margin-right: 14px; text-align: right; color: var(--muted); opacity: .75; font-style: normal; user-select: none; }
details.ct-src pre i::before { content: attr(data-n); }
.ct .p-c, .ct .p-c1, .ct .p-cm, .ct .p-ch, .ct .p-sd { color: var(--muted); }
.ct .p-k, .ct .p-kn, .ct .p-kc, .ct .p-kd, .ct .p-kr, .ct .p-ow, .ct .p-nd { color: var(--accent); font-weight: 500; }
.ct .p-s, .ct .p-s1, .ct .p-s2, .ct .p-sa, .ct .p-sb, .ct .p-se, .ct .p-si, .ct .p-sr { color: var(--ok); }
.ct .p-m, .ct .p-mi, .ct .p-mf, .ct .p-mh, .ct .p-il { color: var(--warn); }
.ct .p-nf, .ct .p-nc, .ct .p-fm { font-weight: 600; }
.cm { display: grid; gap: 8px; min-width: 0; scroll-margin-top: 12px; }
.cm-bar { display: flex; flex-wrap: wrap; gap: 8px 10px; align-items: center; font-size: var(--step--1); }
.cm-bar b { font: 600 var(--step-0)/1.3 var(--f-body); margin-right: 6px; }
.cm-bar button { font: 500 var(--step--1)/1 var(--f-body); padding: 7px 11px; border: 1px solid var(--rule); border-radius: 6px; background: var(--card); color: var(--ink); cursor: pointer; }
.cm-bar button:hover { border-color: var(--accent); color: var(--accent); }
.cm-bar button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.cm-count { color: var(--muted); font-variant-numeric: tabular-nums; }
.cm-legend { color: var(--muted); margin-left: auto; }
.cm-legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--warn); margin-right: 6px; }
.cm-wrap { overflow-x: auto; background: var(--card); border: 1px solid var(--rule); border-radius: 8px; padding: 6px; }
.cm svg { display: block; width: 100%; height: auto; }
.cm-b { fill: var(--card); stroke: var(--ink); stroke-width: 1.2; }
.cm-n.cmd .cm-b { fill: var(--accent-soft); stroke: var(--accent); }
.cm-n.file .cm-b { fill: var(--code-bg); stroke: var(--muted); stroke-dasharray: 4 3; }
.cm-n.mod .cm-b { stroke-width: 1.6; }
.cm-n.ext .cm-b { stroke: var(--muted); stroke-dasharray: 6 3; }
.cm-l { font: 500 11.5px var(--f-mono); fill: var(--ink); }
.cm-s { font: 400 11px var(--f-body); fill: var(--muted); }
.cm-lane { font: 600 11.5px var(--f-body); fill: var(--muted); letter-spacing: .04em; text-transform: uppercase; }
.cm-e { fill: none; stroke: var(--muted); stroke-width: 1.3; }
.cm-ah { fill: var(--muted); }
.cm-dot { fill: var(--warn); }
a.cm-n { cursor: pointer; text-decoration: none; }
.cm-n:focus { outline: none; }
a.cm-n:hover .cm-b, .cm-n:focus .cm-b, .cm-n.cm-on .cm-b { stroke: var(--accent); stroke-width: 2.4; }
svg.cm-dim .cm-n { opacity: .32; } svg.cm-dim .cm-n.cm-on, svg.cm-dim .cm-n.cm-near { opacity: 1; }
svg.cm-dim .cm-e { opacity: .18; } svg.cm-dim .cm-e.cm-on { opacity: 1; stroke: var(--accent); stroke-width: 2; }
.cm-info { min-height: 4.6em; font-size: var(--step--1); max-width: 96ch; }
.cm-info b { font-family: var(--f-mono); font-weight: 500; }
.cm-flagtxt { color: var(--warn); }
.cm-open { white-space: nowrap; font-weight: 600; }
.ct-up { font: 400 var(--step--1)/1.4 var(--f-body); margin-left: auto; }
.ct-seg:target, .ct-file:target > header { outline: 2px solid var(--accent); outline-offset: 8px; border-radius: 4px; }
/* end Code tab */
'''

RADIO = '<input type="radio" name="week" class="tabradio" id="w-tour" value="tour" aria-label="Code">\n'
LABEL = '    <label for="w-tour">Code<small>full source</small></label>\n'
HASH = '      if (/^ct-/.test(h)) { document.getElementById("w-tour").checked = true; return; }\n'


def patch():
    tour, total, nfiles, nseg = build()
    s = SITE.read_text()
    # idempotent: drop an earlier Code tab (the section first, because it quotes this script)
    s = re.sub(r'<section class="week" id="codetour".*?</section>\n', "", s, flags=re.S)
    s = re.sub(r"/\* Code tab \*/.*?/\* end Code tab \*/\n", "", s, count=1, flags=re.S)
    s = s.replace(RADIO, "").replace(LABEL, "").replace(HASH, "")

    def after(anchor, new):
        nonlocal s
        assert s.count(anchor) == 1, (s.count(anchor), anchor[:60])
        s = s.replace(anchor, anchor + new)

    def before(anchor, new):
        nonlocal s
        assert s.count(anchor) == 1, (s.count(anchor), anchor[:60])
        s = s.replace(anchor, new + anchor)

    # styles go ahead of the Walkthrough's block, which that builder deletes up to </style> when it re-runs
    before("/* Walkthrough tab */" if "/* Walkthrough tab */" in s else "</style>", CSS)
    has_walk = 'id="w-walk"' in s
    after('<input type="radio" name="week" class="tabradio" id="w-walk" value="walk" aria-label="Walkthrough">\n' if has_walk else
          '<input type="radio" name="week" class="tabradio" id="w-home" value="home" checked aria-label="Home">\n', RADIO)
    after('    <label for="w-walk">Walkthrough<small>start here</small></label>\n' if has_walk else '    <label for="w-home">Home</label>\n', LABEL)
    before('<section class="week" id="week1" aria-label="Week 1">', tour)
    after('      if (w) { w.checked = true; return; }\n', HASH)
    SITE.write_text(s)
    print(f"code tab: {nfiles} files, {total} lines, {nseg} blocks, {len(tour)} chars; site: {len(s)}")


if __name__ == "__main__":
    patch()
