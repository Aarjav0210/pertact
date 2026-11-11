"""SVG figures for the Walkthrough tab (used by build_walkthrough.py). Each function returns one <svg> string
that uses the site's diagram classes; coordinates are in a 900-wide viewBox."""
import math
import random

C = {"m": "var(--muted)", "a": "var(--accent)", "i": "var(--ink)", "d": "var(--dbl)"}


class Fig:
    def __init__(self, fid, w, h, label, minw=660):
        self.id, self.w, self.h, self.label, self.minw, self.p = fid, w, h, label, minw, []
        self.p.append("<defs>" + "".join(
            f'<marker id="{fid}-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for k, c in C.items()) + "</defs>")

    def raw(self, s):
        self.p.append(s)

    def t(self, x, y, s, cls="d-s", anchor=None, extra=""):
        a = f' text-anchor="{anchor}"' if anchor else ""
        self.p.append(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}"{a}{extra}>{s}</text>')

    def r(self, x, y, w, h, cls="d-box", rx=5, extra=""):
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}"{extra}/>')

    def l(self, pts, c="m", w=1.4, dash=None, arrow=True, op=None):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        e = f' stroke-dasharray="{dash}"' if dash else ""
        e += f' marker-end="url(#{self.id}-{c})"' if arrow else ""
        e += f' stroke-opacity="{op}"' if op else ""
        self.p.append(f'<path d="{d}" fill="none" stroke="{C[c]}" stroke-width="{w}"{e}/>')

    def seg(self, a, b, c="m", w=1.4, cut0=0, cut1=0, **kw):
        """Straight arrow from a to b, trimmed at both ends."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        ux, uy = dx / n, dy / n
        self.l([(a[0] + ux * cut0, a[1] + uy * cut0), (b[0] - ux * cut1, b[1] - uy * cut1)], c, w, **kw)

    def dot(self, x, y, r, fill, extra=""):
        self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"{extra}/>')

    def ten(self, x, y, w, h, inside=None, cls="d-ten", below=None, above=None, icls="d-m"):
        self.r(x, y, w, h, cls, 4)
        if inside:
            self.t(x + w / 2, y + h / 2 + 4.5, inside, icls, "middle")
        if below:
            self.t(x + w / 2, y + h + 16, below, "d-m", "middle")
        if above:
            self.t(x + w / 2, y - 8, above, "d-xs", "middle")

    def op(self, x, y, w, h, l1, l2=None, l3=None):
        self.r(x, y, w, h, "d-op", 6)
        n = 1 + bool(l2) + bool(l3)
        y0 = y + h / 2 + 4.5 - (n - 1) * 9
        self.t(x + w / 2, y0, l1, "d-h", "middle")
        if l2:
            self.t(x + w / 2, y0 + 18, l2, "d-m", "middle")
        if l3:
            self.t(x + w / 2, y0 + 36, l3, "d-m", "middle")

    def svg(self):
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{self.label}" style="min-width:{self.minw}px">\n'
                + "\n".join(self.p) + "\n</svg>")


# ------------------------------------------------------------------ 1. where the lines are
def fig_lines():
    f = Fig("wk0", 900, 226, "Of 214,959 committed lines, 91,261 are the cell split list, 79,326 are result tables, 33,826 are "
            "embedding metadata and 2,931 are Python code.")
    x0, W, tot = 20, 860, 214959
    f.t(x0, 24, "214,959 committed lines", "d-t")
    x = x0
    for name, n, sub in [("splits.json", 91261, "one cell barcode per line"), ("results/", 79326, "metric tables and ranked gene lists"),
                         ("embeddings/", 33826, "one row per gene")]:
        w = n / tot * W
        f.r(x, 40, w - 2, 44, "d-seg", 3)
        f.t(x + 10, 58, name, "d-h")
        f.t(x + 10, 75, f"{n:,}", "d-s")
        f.t(x + 10, 102, sub, "d-xs")
        x += w
    f.r(x, 40, x0 + W - x, 44, "d-seg hot", 3)
    f.t(x0 + W, 32, "everything else: 10,546", "d-xs", "end")
    mid = x + (x0 + W - x) / 2
    f.l([(mid, 88), (mid, 143)])
    f.t(mid - 10, 128, "the last 5%, enlarged below", "d-xs", "end")
    x = x0
    for name, n, sub, code in [("site/index.html", 3990, "this page", False), ("Python: src/ + scripts/", 2931, "9 modules + 5 scripts: all of the code", True),
                               ("logs/", 2506, "run logs", False), ("docs", 1119, "reports, config", False)]:
        w = n / 10546 * W
        f.r(x, 148, w - 2, 44, "d-seg code" if code else "d-seg", 3)
        f.t(x + 10, 166, name, "d-w" if code else "d-h")
        f.t(x + 10, 183, f"{n:,}" + (" lines" if code else ""), "d-wm" if code else "d-s")
        f.t(x + 10, 210, sub, "d-xs")
        x += w
    return f.svg()


# ------------------------------------------------------------------ 1b. module map
def fig_map():
    f = Fig("wk1", 900, 262, "Module map: data.py, splits.py and encoders.py are shared; Week 1 adds embeddings.py, decoders.py, and "
            "metrics.py with runs.py; Week 2 adds ot.py, actions.py with infonce.py, and retrieval.py; decoder_test.py feeds the "
            "action vectors back into the Week 1 decoder.")
    xs, w, h = [130, 370, 610], 190, 48

    def box(x, y, name, desc, cls="d-box"):
        f.r(x, y, w, h, cls, 6)
        f.t(x + 12, y + 20, name, "d-h")
        f.t(x + 12, y + 38, desc, "d-xs")

    for (y, lab), boxes in zip([(22, "Shared"), (112, "Week 1"), (196, "Week 2")],
                               [[("data.py", "QC, normalise, pick genes"), ("splits.py", "train / validation / test"), ("encoders.py", "frozen PCA-50 and scVI-30")],
                                [("embeddings.py", "four gene-embedding tables"), ("decoders.py", "the two decoder models"), ("metrics.py · runs.py", "scoring, baselines, all runs")],
                                [("ot.py", "control-to-cell pairs, via POT"), ("actions.py · infonce.py", "action vectors, their training"), ("retrieval.py", "forward and reverse tests")]]):
        f.t(20, y + 29, lab, "d-t")
        for x, (name, desc) in zip(xs, boxes):
            box(x, y, name, desc, "d-box acc" if lab == "Week 2" else "d-box")
        for a, b in zip(xs[:-1], xs[1:]):
            f.l([(a + w + 3, y + h / 2), (b - 4, y + h / 2)])
    # shared outputs feed both lanes through a bus on the left
    f.l([(705, 70), (705, 92), (104, 92), (104, 220), (126, 220)])
    f.l([(104, 136), (126, 136)])
    f.t(404, 86, "processed matrix, splits and frozen cell states feed both weeks", "d-xs", "middle")
    f.l([(705, 196), (705, 164)], c="a")
    f.t(715, 176, "decoder_test.py:", "d-xs")
    f.t(715, 190, "actions as decoder input", "d-xs")
    return f.svg()


# ------------------------------------------------------------------ 2. the idea
def fig_idea():
    f = Fig("wk2", 900, 318, "Left: an action f(a) is the arrow from the control cells to the cells with gene a switched on. Right: the "
            "composition test adds f(a) and f(b) from a control and compares the landing point with measured cells that carry both.")
    rng = random.Random(7)

    def cloud(cx, cy, n, sd, fill, op=0.8):
        for _ in range(n):
            f.dot(cx + rng.gauss(0, sd), cy + rng.gauss(0, sd * 0.8), 3.2, fill, f' opacity="{op}"')

    f.t(20, 26, "One perturbation is one arrow", "d-t")
    cloud(110, 222, 22, 15, C["m"], 0.55)
    cloud(300, 122, 22, 15, C["a"])
    f.seg((110, 222), (300, 122), "a", 3, 22, 24)
    f.t(190, 158, "f(a)", "d-h", "end")
    f.t(110, 270, "control cells", "d-xs", "middle")
    f.t(300, 84, "cells with gene a switched on", "d-xs", "middle")
    f.t(20, 304, "state(control) + f(a) ≈ state(perturbed)", "d-s")
    f.l([(450, 14), (450, 306)], arrow=False, w=1, op=0.5)
    f.t(474, 26, "The test: do two arrows add up?", "d-t")
    ctrl, A, B, pred, obs = (510, 250), (610, 150), (660, 240), (760, 140), (742, 159)
    cloud(*ctrl, 16, 10, C["m"], 0.55)
    cloud(*A, 16, 10, C["a"])
    cloud(*B, 16, 10, C["a"])
    cloud(*obs, 16, 9, C["d"], 0.75)
    f.seg(ctrl, A, "a", 2.5, 14, 16)
    f.seg(ctrl, B, "a", 2.5, 14, 16)
    f.seg(A, pred, "a", 1.4, 16, 12, dash="5 4")
    f.seg(B, pred, "a", 1.4, 16, 12, dash="5 4")
    f.raw(f'<circle cx="{pred[0]}" cy="{pred[1]}" r="8" fill="none" stroke="{C["i"]}" stroke-width="2"/>')
    f.t(548, 194, "f(a)", "d-h", "end")
    f.t(585, 265, "f(b)", "d-h", "middle")
    f.t(510, 286, "control", "d-xs", "middle")
    f.t(610, 120, "cells with a", "d-xs", "middle")
    f.t(672, 272, "cells with b", "d-xs", "middle")
    f.t(760, 112, "predicted: f(a) + f(b)", "d-xs", "middle")
    f.t(776, 184, "measured a + b", "d-xs")
    f.t(474, 304, "prediction for a + b = state(control) + f(a) + f(b)", "d-s")
    return f.svg()


# ------------------------------------------------------------------ 3. cell filter funnel
def fig_funnel():
    f = Fig("wk3", 900, 318, "Cell filtering: 111,445 cells in the file; 104,507 with a confident guide call; 91,168 with exactly one "
            "guide identity; 91,056 after the mitochondrial filter; 90,867 after dropping perturbations with fewer than 50 cells.")
    x0, sc = 320, 420 / 111445
    f.t(20, 22, "Cells kept after each filter", "d-t")
    for y, lab, n, note in [(40, "scPerturb file, as downloaded", 111445, ""), (84, "guide call has good coverage", 104507, "− 6,938"),
                            (128, "exactly one guide identity", 91168, "− 13,339 multiplets"), (172, "mitochondrial reads ≤ 15%", 91056, "− 112"),
                            (216, "perturbation has ≥ 50 cells", 90867, "− 189 (4 doubles)")]:
        f.t(20, y + 18, lab, "d-h")
        f.r(x0, y, n * sc, 26, "d-seg", 3)
        f.t(x0 + n * sc + 8, y + 18, f"{n:,}", "d-m")
        if note:
            f.t(880, y + 18, note, "d-xs", "end")
    y, x = 262, x0
    f.t(20, y + 18, "what is left: 90,867 cells", "d-h")
    for n, fill, op in [(9687, C["m"], 0.55), (47214, C["a"], 1), (33966, C["d"], 0.85)]:
        f.raw(f'<rect x="{x:.1f}" y="{y}" width="{n * sc - 2:.1f}" height="26" rx="3" fill="{fill}" opacity="{op}"/>')
        x += n * sc
    lx = 320
    for lab, fill, op, wd in [("9,687 control", C["m"], 0.55, 120), ("47,214 single (105 genes)", C["a"], 1, 190), ("33,966 double (127 pairs)", C["d"], 0.85, 0)]:
        f.raw(f'<rect x="{lx}" y="298" width="10" height="10" rx="2" fill="{fill}" opacity="{op}"/>')
        f.t(lx + 16, 307, lab, "d-xs")
        lx += wd
    return f.svg()


# ------------------------------------------------------------------ 4. split
def fig_split():
    f = Fig("wk4", 900, 330, "The split. Single-perturbation cells: 37,770 train, 9,444 validation, none in test. Controls: 6,781, "
            "1,453 and 1,453. Double-perturbation cells: none in train or validation, all 33,966 in test.")
    cols = [(300, "train", "44,551 cells"), (500, "validation", "10,897 cells"), (700, "test", "35,419 cells")]
    for x, name, tot in cols:
        f.t(x, 28, name, "d-t")
        f.t(x + 180, 28, tot, "d-s", "end")
    rows = [(46, "105 single perturbations", "47,214 cells, split 80 / 20 per perturbation", [37770, 9444, 0], "d-fill-acc"),
            (116, "Control cells", "9,687 cells, split 70 / 15 / 15 per gem group", [6781, 1453, 1453], "d-fill-ctl"),
            (186, "127 double perturbations", "33,966 cells, all held out", [0, 0, 33966], "d-fill-dbl")]
    for y, l1, l2, ns, cls in rows:
        f.t(20, y + 25, l1, "d-h")
        f.t(20, y + 43, l2, "d-xs")
        for (x, _, _), n in zip(cols, ns):
            if n:
                f.r(x, y, 180, 56, "d-box", 5)
                f.r(x, y, max(5, n / 47214 * 180), 56, cls, 5)
                f.t(x + 12, y + 34, f"{n:,}", "d-t")
            else:
                f.r(x, y, 180, 56, "d-box none", 5)
                f.t(x + 12, y + 26, "0", "d-s")
                f.t(x + 12, y + 43, "never", "d-xs")
    f.l([(20, 258), (880, 258)], arrow=False, w=1, op=0.5)
    f.t(20, 284, "Used for", "d-h")
    for (x, _, _), (a, b) in zip(cols, [("fitting everything: genes,", "encoders, pairs, actions, decoders"), ("early stopping, tuning,", "every Week 2 score"),
                                        ("Week 1 final scores only,", "untouched in Week 2")]):
        f.t(x, 284, a, "d-xs")
        f.t(x, 301, b, "d-xs")
    return f.svg()


# ------------------------------------------------------------------ 5. tensors from counts to state
def fig_tensors():
    f = Fig("wk5", 900, 392, "Tensor shapes through preprocessing: raw counts 111,445 by 33,694; after cell QC 90,867 by 33,694; expressed "
            "genes 90,867 by 17,285; model matrix X 90,867 by 5,028; then PCA to 90,867 by 50 and scVI to 90,867 by 30, both fitted on training cells.")
    f.t(20, 24, "Rows are cells, columns are genes", "d-t")
    T = [(20, 150, 100, "raw counts", "111,445 × 33,694", "d-ten ctl"), (296, 150, 82, "after cell QC", "90,867 × 33,694", "d-ten ctl"),
         (572, 110, 82, "expressed genes", "90,867 × 17,285", "d-ten ctl"), (808, 72, 82, "X: model genes", "90,867 × 5,028", "d-ten")]
    for x, w, h, name, shape, cls in T:
        y = 110 - h / 2
        f.r(x, y, w, h, cls, 4)
        f.t(x + w / 2, y - 9, name, "d-h", "middle")
        f.t(x + w / 2, y + h + 17, shape, "d-m", "middle")
    for (x, w, *_), (x2, *_), up, dn, fit in zip(T[:-1], T[1:], ["cell QC", "log1p(CP10k)", "5,000 HVGs"], ["− 20,578 cells", "≥ 20 train cells", "+ 28 target genes"],
                                                 [None, "train cells only", "train cells only"]):
        f.l([(x + w + 4, 110), (x2 - 5, 110)])
        mx = (x + w + x2) / 2
        f.t(mx, 101, up, "d-xs", "middle")
        f.t(mx, 127, dn, "d-xs", "middle")
        if fit:
            f.t(mx, 143, fit, "d-ok", "middle")
    f.l([(20, 208), (880, 208)], arrow=False, w=1, op=0.5)
    f.t(20, 234, "Two frozen encoders turn each cell's 5,028 genes into a short state vector", "d-t")
    f.r(20, 256, 72, 112, "d-ten", 4)
    f.t(56, 316, "X", "d-h", "middle")
    for y, l1, l2, tw, name, use in [(284, "PCA, 50 components", "fit on 44,551 train cells", 24, "S_pca   90,867 × 50", "decoder state input, OT pairing, actions"),
                                     (340, "scVI, 30 latent dims", "trained on train-cell counts", 16, "S_scvi  90,867 × 30", "OT pairing, actions (a 10-dim one is built, unused)")]:
        f.l([(96, y), (146, y)])
        f.r(150, y - 22, 270, 44, "d-op", 6)
        f.t(285, y - 3, l1, "d-h", "middle")
        f.t(285, y + 13, l2, "d-ok", "middle")
        f.l([(424, y), (476, y)])
        f.r(480, y - 22, tw, 44, "d-ten", 3)
        f.t(520, y - 3, name, "d-m")
        f.t(520, y + 14, use, "d-xs")
    return f.svg()


# ------------------------------------------------------------------ 7. decoder
def fig_decoder():
    f = Fig("wk6", 900, 372, "Latent additive decoder with tensor shapes: control state B by 50 goes through an encoder MLP to B by 128; the "
            "gene embedding B by d goes through a bias-free linear map W to B by 128; the two are added; a decoder MLP outputs B by 5,028, "
            "trained with mean squared error against a real perturbed cell.")
    f.t(20, 28, "One training batch, B = 256", "d-t")
    f.r(10, 50, 536, 108, "d-out", 8)
    f.t(540, 44, "decoder-only variant: this whole branch is removed", "d-note", "end")
    f.t(20, 72, "control state, PCA-50", "d-xs")
    f.ten(20, 80, 92, 44, "s_ctrl", "d-ten ctl", icls="d-h")
    f.t(66, 142, "[B, 50]", "d-m", "middle")
    f.l([(116, 102), (156, 102)])
    f.op(160, 74, 250, 56, "enc: MLP with 2 hidden layers", "50 → 1024 → 1024 → 128")
    f.t(285, 148, "1.24 M weights, LayerNorm + ReLU", "d-xs", "middle")
    f.l([(414, 102), (456, 102)])
    f.ten(460, 84, 72, 36, "[B, 128]")
    f.t(20, 214, "embedding row of gene a", "d-xs")
    f.ten(20, 222, 92, 44, "e(a)", icls="d-h")
    f.t(66, 284, "[B, d]", "d-m", "middle")
    f.t(20, 308, "a double: e(a) + e(b)", "d-xs")
    f.t(20, 325, "a control cell: e = 0", "d-xs")
    f.l([(116, 244), (156, 244)])
    f.op(160, 216, 250, 56, "W: one linear layer, no bias", "d → 128")
    f.t(285, 290, "d × 128 weights (13,440 for one-hot)", "d-xs", "middle")
    f.l([(414, 244), (456, 244)])
    f.ten(460, 226, 72, 36, "[B, 128]")
    f.raw(f'<circle cx="590" cy="173" r="15" class="d-op"/>')
    f.t(590, 178.5, "+", "d-t", "middle")
    f.seg((532, 102), (590, 173), cut0=2, cut1=19)
    f.seg((532, 244), (590, 173), cut0=2, cut1=19)
    f.l([(606, 173), (626, 173)])
    f.ten(630, 155, 72, 36, "[B, 128]")
    f.t(666, 147, "z", "d-h", "middle")
    f.l([(704, 173), (728, 173)])
    f.op(732, 133, 148, 80, "dec: MLP", "128 → 1024", "→ 1024 → 5,028")
    f.t(880, 125, "6.34 M weights", "d-xs", "end")
    f.l([(806, 214), (806, 295)])
    f.ten(746, 300, 120, 40, "x̂  [B, 5,028]")
    f.l([(744, 320), (692, 320)])
    f.op(540, 298, 148, 44, "MSE loss", None)
    f.l([(484, 320), (536, 320)])
    f.ten(300, 300, 180, 40, "x  [B, 5,028]")
    f.t(390, 358, "a real training cell: perturbed with a, or a control", "d-xs", "middle")
    return f.svg()


# ------------------------------------------------------------------ 8a. metric
def fig_metric():
    f = Fig("wk7", 900, 206, "Scoring one condition: average 500 decoder outputs, average the measured cells, subtract the control mean from "
            "both, keep the condition's 20 most-changed genes, and take the Pearson correlation of the two 20-number vectors.")
    rows = [(56, "Decoder output", "for 500 control inputs", "[500, 5,028]", "x̂ [5,028]", "d-ten"),
            (116, "Measured cells", "of the same condition", "[n, 5,028]", "x [5,028]", "d-ten"),
            (176, "Control cells", "of the same split", "[m, 5,028]", "c [5,028]", "d-ten ctl")]
    for y, l1, l2, s1, s2, cls in rows:
        f.t(20, y - 2, l1, "d-h")
        f.t(20, y + 14, l2, "d-xs")
        f.ten(206, y - 15, 124, 30, s1, cls)
        f.l([(334, y), (388, y)])
        f.t(361, y - 7, "mean", "d-xs", "middle")
        f.ten(392, y - 15, 96, 30, s2, cls)
    for y, s in [(56, "Δ̂ [5,028]"), (116, "Δ [5,028]")]:
        f.l([(492, y), (556, y)])
        f.t(532, y - 7, "− c", "d-xs", "middle")
        f.ten(560, y - 15, 96, 30, s)
        f.l([(660, y), (730, y)])
        f.t(695, y - 7, "top-20 DE", "d-xs", "middle")
        f.ten(734, y - 15, 56, 30, "[20]")
        f.dot(512, y, 2.6, C["m"])
    f.l([(492, 176), (512, 176), (512, 56)], arrow=False)
    f.op(812, 64, 76, 44, "Pearson r")
    f.seg((790, 56), (812, 76), cut0=2, cut1=3)
    f.seg((790, 116), (812, 96), cut0=2, cut1=3)
    f.t(560, 172, "score = how well predicted and measured", "d-xs")
    f.t(560, 188, "change agree on the 20 genes that moved most", "d-xs")
    return f.svg()


# ------------------------------------------------------------------ 8b. week 1 results
def fig_w1():
    f = Fig("wk8", 900, 326, "Week 1 test result on 127 held-out double perturbations, Pearson correlation on the top 20 changed genes. "
            "Adding the two singles with no model scores 0.968. One-hot scores 0.960 and 0.963, GenePT 0.950 and 0.954, PCA loadings 0.923 "
            "and 0.916, STRING on its 108 covered pairs 0.934 and 0.951, STRING on all 127 pairs 0.887 and 0.895.")
    x0, v0, k = 250, 0.86, 530 / 0.12
    X = lambda v: x0 + (v - v0) * k
    f.raw(f'<circle cx="256" cy="20" r="5.5" fill="var(--s1)"/>')
    f.t(268, 24, "latent additive", "d-xs")
    f.raw(f'<circle cx="372" cy="20" r="5.5" fill="var(--s2)"/>')
    f.t(384, 24, "decoder-only", "d-xs")
    f.raw(f'<path d="M474,14 l6,6 l-6,6 l-6,-6 z" fill="{C["i"]}"/>')
    f.t(486, 24, "no model", "d-xs")
    f.l([(X(0.968), 62), (X(0.968), 286)], arrow=False, dash="3 3", w=1)
    f.t(X(0.968) - 6, 54, "0.968: just add the two singles", "d-xs", "end")
    f.t(880, 54, "scores", "d-xs", "end")
    rows = [(78, "Adding the two singles, no model", None, None, 0.968), (118, "One-hot", 0.960, 0.963, None), (158, "GenePT (text)", 0.950, 0.954, None),
            (198, "PCA loadings", 0.923, 0.916, None), (238, "STRING, its 108 covered pairs", 0.934, 0.951, None), (278, "STRING, all 127 pairs", 0.887, 0.895, None)]
    for y, lab, a, b, base in rows:
        f.l([(x0, y), (780, y)], arrow=False, w=1, op=0.35)
        f.t(20, y + 4, lab, "d-h")
        if base:
            f.raw(f'<g><title>{lab}: {base:.3f}</title><path d="M{X(base):.1f},{y - 7} l7,7 l-7,7 l-7,-7 z" fill="{C["i"]}"/></g>')
            f.t(880, y + 4, f"{base:.3f}", "d-m", "end")
        else:
            for v, col, nm in [(a, "var(--s1)", "latent additive"), (b, "var(--s2)", "decoder-only")]:
                f.raw(f'<g><title>{lab}, {nm}: {v:.3f}</title><circle cx="{X(v):.1f}" cy="{y}" r="11" fill="transparent"/>'
                      f'<circle cx="{X(v):.1f}" cy="{y}" r="5.5" fill="{col}" stroke="#fff" stroke-width="2"/></g>')
            f.t(880, y + 4, f"{a:.3f} / {b:.3f}", "d-m", "end")
    f.l([(x0, 292), (780, 292)], arrow=False, w=1)
    for v in [0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]:
        f.l([(X(v), 292), (X(v), 297)], arrow=False, w=1)
        f.t(X(v), 311, f"{v:.2f}", "d-xs", "middle")
    return f.svg()


# ------------------------------------------------------------------ 9. OT
def fig_ot():
    f = Fig("wk9", 900, 292, "One OT problem: 861 control states and 41 perturbed-cell states, each 50 numbers, give a cost matrix of 861 "
            "by 41 squared distances; Sinkhorn-Knopp turns it into a plan P of the same shape; the 5 heaviest controls in each column are kept "
            "as partners with weights, each 41 by 5.")
    f.t(20, 24, "One pairing problem: one perturbation in one gem group (median training sizes)", "d-t")
    f.ten(20, 62, 96, 60, "X_c", "d-ten ctl", "[861, 50]", "controls, one gem group", "d-h")
    f.ten(20, 190, 96, 34, "X_p", "d-ten", "[41, 50]", "cells perturbed with a", "d-h")
    f.seg((116, 92), (190, 122), cut0=3, cut1=4)
    f.seg((116, 207), (190, 158), cut0=3, cut1=4)
    f.ten(190, 98, 96, 84, "C", "d-ten ctl", "[861, 41]", "squared distance, every pair", "d-h")
    f.l([(290, 140), (336, 140)])
    f.r(340, 104, 176, 72, "d-op", 6)
    f.t(428, 127, "Sinkhorn-Knopp", "d-h", "middle")
    f.t(428, 146, "blur ε = 0.05 × median", "d-xs", "middle")
    f.t(428, 162, "control distance²", "d-xs", "middle")
    f.l([(520, 140), (566, 140)])
    gx, gy, cs = 570, 84, 14
    for j in range(10):
        main = (j * 3) % 8
        for i in range(8):
            op = {0: 0.92, 1: 0.45, 5: 0.22}.get((i - main) % 8, 0.05)
            f.raw(f'<rect x="{gx + j * cs}" y="{gy + i * cs}" width="{cs - 1.5}" height="{cs - 1.5}" rx="2" fill="{C["a"]}" opacity="{op}"/>')
    f.t(gx + 70, 72, "plan P  [861, 41]", "d-m", "middle")
    f.t(gx + 70, 214, "rows: controls, columns: perturbed cells", "d-xs", "middle")
    f.t(gx + 70, 230, "every row sums to 1/861, every column to 1/41", "d-xs", "middle")
    f.l([(714, 140), (758, 140)], arrow=False)
    f.l([(758, 140), (758, 112), (772, 112)])
    f.l([(758, 140), (758, 172), (772, 172)])
    f.ten(776, 98, 104, 28, "[41, 5]", above="5 heaviest controls")
    f.ten(776, 158, 104, 28, "[41, 5]", above="their weights, sum 1")
    f.t(20, 268, "840 such problems per state space for training cells (105 perturbations × 8 gem groups), 834 for validation cells. "
        , "d-xs")
    return f.svg()


# ------------------------------------------------------------------ 10. InfoNCE
def fig_infonce():
    f = Fig("wk10", 900, 432, "One InfoNCE step with tensor shapes: the action table F, 105 by 50, is indexed by each pair's gene to give "
            "256 by 50; added to the control states it gives queries q, 256 by 50; every query is scored against every perturbed cell to give "
            "logits L, 256 by 256; cross-entropy pushes each row and column toward its diagonal entry. Only F is trained. The mean shift "
            "instead averages 37,770 pair displacements per gene.")
    f.t(20, 26, "One InfoNCE training step: a batch of 256 OT pairs", "d-t")
    f.t(20, 58, "action table: the only trainable weights", "d-xs")
    f.ten(20, 64, 128, 50, "F  [105, 50]", "d-ten strong")
    f.t(20, 146, "gene index of each pair", "d-xs")
    f.ten(20, 152, 128, 26, "a  [256]", "d-ten ctl")
    f.t(20, 216, "OT-partner controls", "d-xs")
    f.ten(20, 222, 128, 30, "s_ctrl [256, 50]", "d-ten ctl")
    f.t(20, 316, "perturbed cells: 32 genes × 8 cells", "d-xs")
    f.ten(20, 322, 128, 30, "s_pert [256, 50]")
    f.seg((148, 89), (236, 126), cut0=3, cut1=4)
    f.seg((148, 165), (236, 140), cut0=3, cut1=4)
    f.ten(236, 118, 112, 30, "F[a] [256, 50]", above="each pair's action")
    f.raw(f'<circle cx="410" cy="186" r="14" class="d-op"/>')
    f.t(410, 191.5, "+", "d-t", "middle")
    f.seg((348, 133), (410, 186), cut0=3, cut1=18)
    f.seg((148, 237), (410, 186), cut0=3, cut1=18)
    f.l([(425, 186), (448, 186)])
    f.ten(452, 171, 112, 30, "q [256, 50]", above="query = control + its action")
    f.l([(568, 186), (592, 186)])
    f.op(596, 160, 144, 52, "score every pair", "−‖q − k‖² / (τ·D₀)")
    f.l([(152, 337), (668, 337), (668, 216)])
    f.t(400, 330, "keys k = the batch's perturbed cells", "d-xs", "middle")
    f.l([(744, 186), (766, 186)])
    mx, my, ms = 770, 126, 110
    f.r(mx, my, ms, ms, "d-seg", 3)
    for i in range(4):
        f.raw(f'<rect x="{mx + i * ms / 4 + 1:.1f}" y="{my + i * ms / 4 + 1:.1f}" width="{ms / 4 - 2:.1f}" height="{ms / 4 - 2:.1f}" rx="2" fill="{C["d"]}" opacity="0.32"/>')
    f.raw(f'<path d="M{mx + 2},{my + 2} L{mx + ms - 2},{my + ms - 2}" stroke="{C["a"]}" stroke-width="3.5" stroke-linecap="round"/>')
    f.t(mx + ms / 2, 118, "L  [256, 256]", "d-m", "middle")
    f.l([(825, 240), (825, 264)])
    f.op(752, 268, 146, 46, "cross-entropy")
    f.t(825, 330, "target: the diagonal", "d-xs", "middle")
    f.t(825, 346, "gradient reaches F only", "d-note", "middle")
    f.l([(20, 370), (880, 370)], arrow=False, w=1, op=0.5)
    f.t(20, 399, "Mean shift, for contrast", "d-h")
    f.ten(210, 380, 170, 30, "disp [37,770, 50]", "d-ten ctl")
    f.t(295, 426, "s_pert − Σ w·s_ctrl for every training pair", "d-xs", "middle")
    f.l([(384, 395), (466, 395)])
    f.t(425, 388, "mean per gene", "d-xs", "middle")
    f.ten(470, 380, 140, 30, "F_ms [105, 50]")
    f.t(626, 399, "closed form: no training, no τ", "d-xs")
    return f.svg()


# ------------------------------------------------------------------ 11. three tests
def fig_tests():
    f = Fig("wk11", 900, 280, "Three tests of an action vector. Find the cell: add f(a) to a control and rank held-out cells by distance. "
            "Name the perturbation: from one control try all 105 actions and see which lands nearest the cell. Feed a decoder: use f(a) as the "
            "input of a network that predicts gene-expression change.")
    for k, (title, sub, foot) in enumerate([("Find the cell", "control + f(a): which cell is nearest?", "rank 9,444 cells → MRR"),
                                            ("Name the perturbation", "try all 105 actions from one control", "nearest = the right one? → top-1"),
                                            ("Feed a decoder", "the action is a network's input", "vs measured change → Pearson r")]):
        ox = 20 + 290 * k
        f.r(ox, 12, 280, 256, "d-box", 8)
        f.t(ox + 140, 38, title, "d-t", "middle")
        f.t(ox + 140, 56, sub, "d-xs", "middle")
        f.t(ox + 140, 252, foot, "d-s", "middle")
        if k == 0:
            for x, y in [(75, 120), (120, 86), (235, 140), (215, 190), (140, 178), (240, 88)]:
                f.dot(ox + x, y, 4, C["m"], ' opacity="0.5"')
            f.seg((ox + 40, 200), (ox + 176, 112), "a", 2.5, 8, 10)
            f.raw(f'<circle cx="{ox + 180}" cy="109" r="7" fill="none" stroke="{C["i"]}" stroke-width="2"/>')
            f.dot(ox + 198, 100, 5, C["a"])
            f.dot(ox + 40, 200, 5, C["m"])
            f.t(ox + 40, 222, "control", "d-xs", "middle")
        elif k == 1:
            for x, y in [(85, 88), (140, 78), (238, 160), (135, 132)]:
                f.seg((ox + 40, 200), (ox + x, y), "a", 1.2, 8, 4, op=0.4)
            f.seg((ox + 40, 200), (ox + 204, 106), "a", 2.8, 8, 6)
            f.dot(ox + 222, 98, 5, C["a"])
            f.dot(ox + 40, 200, 5, C["m"])
            f.t(ox + 40, 222, "control", "d-xs", "middle")
        else:
            f.ten(ox + 24, 92, 70, 40, "f(a)", "d-ten strong", icls="d-h")
            f.l([(ox + 98, 112), (ox + 126, 112)])
            f.op(ox + 130, 88, 110, 48, "decoder")
            f.l([(ox + 185, 140), (ox + 185, 158)])
            f.l([(ox + 118, 196), (ox + 252, 196)], arrow=False, w=1)
            for i, hgt in enumerate([18, -12, 26, 8, -16, 12, 20]):
                y = 196 - hgt if hgt > 0 else 196
                f.raw(f'<rect x="{ox + 124 + i * 18}" y="{y}" width="11" height="{abs(hgt)}" rx="2" fill="{C["a"]}" opacity="0.55"/>')
            f.t(ox + 185, 230, "predicted gene changes", "d-xs", "middle")
    return f.svg()


if __name__ == "__main__":
    for fn in [fig_lines, fig_map, fig_idea, fig_funnel, fig_split, fig_tensors, fig_decoder, fig_metric, fig_w1, fig_ot, fig_infonce, fig_tests]:
        print(fn.__name__, len(fn()))
