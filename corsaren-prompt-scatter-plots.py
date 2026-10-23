import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import diptest
import torch
import open_clip
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from scipy.stats import spearmanr

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# ── Data ── #
HF_REPO          = "chcaa/corsaren-caricatures"
NAME_COLUMN      = "filename"
IMAGE_COLUMN     = "image_bytes"
EMBEDDING_COLUMN = "eva02_clip"
TEXT_MODEL       = ("EVA02-L-14", "merged2b_s4b_b131k")

# ── Measures ── #
LIFT_SHARE  = 0.10      					# lift: share of top images per term (0.10 = top 10%); max lift = 1 / share
POLES_FILE        = BASE_DIR / "poles.json" # POLES_FILE includes handpicked images
IEAT_POLE_A       = "dog" 					# if not found in POLE_FILE, will be used as prompt instead
IEAT_POLE_B       = "person"				# if not found in POLE_FILE, will be used as prompt instead
IEAT_PERMUTATIONS = 10000

# ── Plot ── #
SHOW_PLOT      = True   # True = draw the scatter plot after each pair; False = table only
THUMB_SIZE     = 70     # pixels for scatter thumbnails
PLOT_BG        = "#ffffff"
GRID_COLOR     = "#e6e6ea"
TEXT_COLOR     = "#222222"
LINE_COLOR     = "#9a9aa8"
FRAME_COLOR    = "#c8c8d0"   # thin frame around each thumbnail
TERM1_COLOR    = "#b56340"   # x-axis term
TERM2_COLOR    = "#0082cb"   # y-axis term

# ── Table (printed in the terminal, and each row appended to RESULTS_FILE) ── #
TABLE_COLUMNS = ["term1", "term2", "top_n", "text_sim", "rho", "shared", "lift", "dip_p",
				 "ieat_d", "ieat_p", "lean_shared", "poles"]
RESULTS_FILE = BASE_DIR / "clip_scatter_results.csv"
DECIMAL_SEP  = "."      					# "," for Excel with comma decimals (columns then split by ";")

# =============================================================================
# FUNCTIONALITY
# =============================================================================

# ── Loading ── #

def load_data():
	"""Normalised embeddings (dot product = cosine similarity), filenames,
	and the image bytes (only loaded if the plot is shown).
	"""
	from huggingface_hub import hf_hub_download
	print(f"Loading {HF_REPO} ...")
	file = hf_hub_download(HF_REPO, "corsaren-caricatures.parquet", repo_type="dataset")

	columns = [NAME_COLUMN, EMBEDDING_COLUMN] + ([IMAGE_COLUMN] if SHOW_PLOT else [])
	df = pd.read_parquet(file, columns=columns)
	emb = np.stack(df[EMBEDDING_COLUMN]).astype(np.float32)
	# Normalise to length 1, so a dot product is a cosine similarity.
	emb /= np.linalg.norm(emb, axis=1, keepdims=True)
	images = df[IMAGE_COLUMN].tolist() if SHOW_PLOT else None
	print(f"  {len(df)} images, embedding size {emb.shape[1]}\n")
	return emb, df[NAME_COLUMN].tolist(), images


def load_text_model():
	"""The CLIP text encoder and its tokenizer."""
	name, pretrained = TEXT_MODEL
	print(f"Loading text model {name} ...")
	model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
	model.visual = None                     # the image half isn't needed: frees memory
	return model.eval(), open_clip.get_tokenizer(name)


def encode_texts(texts, text_model):
	"""Texts as normalised CLIP embeddings, one row per text."""
	model, tokenizer = text_model
	with torch.no_grad():
		t = model.encode_text(tokenizer(texts))
	return (t / t.norm(dim=-1, keepdim=True)).numpy()


# ── Measures ── #


def separability(x, y, t1, t2, top_x, top_y):
	"""How differently the two terms sort the images.

	x, y: every image's similarity to term 1 and term 2; t1, t2: the two text
	embeddings; top_x, top_y: the top N images of each term.
	The dip test only uses images in the top N of at least one term.
	"""
	# Lift: images in the top LIFT_SHARE for both terms, relative to chance
	lift = ((x >= np.quantile(x, 1 - LIFT_SHARE)) &
			(y >= np.quantile(y, 1 - LIFT_SHARE))).mean() / LIFT_SHARE ** 2

	# Dip test on the difference of standardised scores (>0 = more term 1)
	z = lambda a: (a - a.mean()) / a.std()
	d = (z(x) - z(y))[np.union1d(top_x, top_y)]

	return {
		"text_sim": float(t1 @ t2),
		"rho":      float(spearmanr(x, y)[0]),
		"shared":   len(np.intersect1d(top_x, top_y)),
		"lift":     float(lift),
		"dip_p":    float(diptest.diptest(d)[1]),
	}


def pole_sets(labels, emb, ta, tb, n):
	"""The two iEAT pole image sets and a description for the table.
	"""
	saved = json.loads(POLES_FILE.read_text(encoding="utf-8")) if POLES_FILE.exists() else {}
	index_of = {name: i for i, name in enumerate(labels)}

	def one(name, t):
		if name in saved:
			return {index_of[f] for f in saved[name]}, f"{name} (manual)"
		return set(np.argsort(-(emb @ t))[:n].tolist()), f"{name} (prompt)"

	(A, desc_a), (B, desc_b) = one(IEAT_POLE_A, ta), one(IEAT_POLE_B, tb)
	return A, B, f"{desc_a} vs {desc_b}"


def ieat(emb, top_x, top_y, A, B):
	"""Image Embedding Association Test (Steed & Caliskan 2021).

	Targets X, Y: the top N images of term 1 and 2; poles A, B.
	Per target image, s = mean similarity to A minus mean similarity to B.
	d: (mean s of X - mean s of Y) / std of s, like Cohen's d; d > 0 means
	term 1's images lean more towards A than term 2's.
	p: two-sided permutation test (shuffling X/Y labels).
	lean_shared: position of the shared top-N images on the same scale, in
	standard deviations (> 0 = towards A); not part of the test.
	"""
	X, Y = set(top_x.tolist()), set(top_y.tolist())
	S = X & Y									# shared images, kept aside for lean_shared
	X, Y = X - Y, Y - X							# drop images shared by the targets
	A, B = A - B, B - A							# and by the poles
	A, B = A - (X | Y), B - (X | Y)				# targets take precedence over poles
	if min(len(X), len(Y), len(A), len(B)) < 2:
		return {"ieat_d": np.nan, "ieat_p": np.nan, "lean_shared": np.nan}   # shown as empty

	X, Y, A, B = (np.array(sorted(group)) for group in (X, Y, A, B))
	W = np.concatenate([X, Y])
	s = (emb[W] @ emb[A].T).mean(axis=1) - (emb[W] @ emb[B].T).mean(axis=1)
	nx = len(X)
	diff = s[:nx].mean() - s[nx:].mean()

	rng = np.random.default_rng(0)				# fixed seed: same p for the same input
	perm = np.empty(IEAT_PERMUTATIONS)
	for i in range(IEAT_PERMUTATIONS):
		idx = rng.permutation(len(s))
		perm[i] = s[idx[:nx]].mean() - s[idx[nx:]].mean()
	p = (np.sum(np.abs(perm) >= abs(diff)) + 1) / (IEAT_PERMUTATIONS + 1)

	centre, sd = s.mean(), s.std(ddof=1)
	lean_shared = np.nan
	if S:
		S = np.array(sorted(S))
		sim_a = emb[S] @ emb[A].T
		sim_b = emb[S] @ emb[B].T
		sim_a[S[:, None] == A[None, :]] = np.nan	# ignore an image's similarity to itself
		sim_b[S[:, None] == B[None, :]] = np.nan
		s_shared = np.nanmean(sim_a, axis=1) - np.nanmean(sim_b, axis=1)
		lean_shared = float((s_shared.mean() - centre) / sd)

	return {"ieat_d": float(diff / sd), "ieat_p": float(p), "lean_shared": lean_shared}


# Short numbers for the terminal (the CSV keeps 4 decimals)
_num = lambda v: f"{v:.2f}"
_p   = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"
TERMINAL_FORMAT = {"text_sim": _num, "rho": _num, "lift": _num, "dip_p": _p,
				   "ieat_d": _num, "ieat_p": _p, "lean_shared": _num}


def save_row(m):
	"""Append one result row to RESULTS_FILE (header only when the file is new)."""
	pd.DataFrame([m])[TABLE_COLUMNS].to_csv(
		RESULTS_FILE, mode="a", header=not RESULTS_FILE.exists(), index=False,
		sep=";" if DECIMAL_SEP == "," else ",", decimal=DECIMAL_SEP, float_format="%.4f")


# =============================================================================
# TERMINAL INTERACTION / PLOTTING
# =============================================================================

def load_image(image, max_size):
	img = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
	img.thumbnail((max_size, max_size), Image.LANCZOS)
	return np.array(img)


def make_plot(x, y, top_x, top_y, images, labels, term1, term2):
	"""Thumbnails of both terms' top N images at their two similarity scores."""
	above = np.union1d(top_x, top_y)
	cut_x, cut_y = x[top_x[-1]], y[top_y[-1]]		# score of each term's N-th image
	print(f"Rendering {len(above)} thumbnails ...")

	fig, ax = plt.subplots(figsize=(11, 9), facecolor=PLOT_BG)
	ax.set_facecolor(PLOT_BG)
	ax.grid(color=GRID_COLOR, linewidth=0.5, zorder=0)

	# Cut-off lines
	ax.axvline(cut_x, color=TERM1_COLOR, alpha=0.6, linewidth=1, linestyle="--", zorder=1)
	ax.axhline(cut_y, color=TERM2_COLOR, alpha=0.6, linewidth=1, linestyle="--", zorder=1)

	# Invisible points so the axes get their range before the diagonal is drawn
	ax.scatter(x[above], y[above], s=0, zorder=2)

	# Diagonal (y = x): separates "more term 1" from "more term 2"
	lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
	hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
	ax.plot([lo, hi], [lo, hi], color=LINE_COLOR, linewidth=1.2, zorder=2)
	ax.text(0.98, 0.02, f'more "{term1}"', transform=ax.transAxes,
			color=TERM1_COLOR, fontsize=8, ha="right", va="bottom", zorder=3)
	ax.text(0.02, 0.98, f'more "{term2}"', transform=ax.transAxes,
			color=TERM2_COLOR, fontsize=8, ha="left", va="top", zorder=3)

	# Thumbnails
	boxes = []
	for idx in above:
		thumb = load_image(images[idx], THUMB_SIZE)
		ab = AnnotationBbox(OffsetImage(thumb, zoom=THUMB_SIZE / 200), (x[idx], y[idx]),
							frameon=True, pad=0.1, zorder=4,
							bboxprops=dict(edgecolor=FRAME_COLOR, linewidth=0.5,
										   facecolor="none"))
		ax.add_artist(ab)
		boxes.append((ab, idx))

	for spine in ax.spines.values():
		spine.set_edgecolor(GRID_COLOR)
	ax.tick_params(colors=TEXT_COLOR, labelsize=9)
	ax.set_xlabel(f'<- similarity to: "{term1}"', color=TERM1_COLOR, fontsize=11, labelpad=10)
	ax.set_ylabel(f'<- similarity to: "{term2}"', color=TERM2_COLOR, fontsize=11, labelpad=10)
	ax.set_title(f"CLIP Scatter  |  top {len(top_x)} per term  |  {len(above)} images  |  "
				 f"click to enlarge", color=TEXT_COLOR, fontsize=12, pad=14, fontweight="bold")

	def on_click(event):
		"""Open the thumbnail closest to the click in its own window."""
		if event.inaxes is None:
			return
		_, idx = min(boxes, key=lambda b: (b[0].xy[0] - event.xdata) ** 2 +
										  (b[0].xy[1] - event.ydata) ** 2)
		popup, pax = plt.subplots(figsize=(5, 5), facecolor=PLOT_BG)
		popup.canvas.manager.set_window_title(labels[idx])
		pax.imshow(load_image(images[idx], 600))
		pax.axis("off")
		pax.set_title(labels[idx], color=TEXT_COLOR, fontsize=10, pad=8)
		popup.tight_layout()
		popup.show()
	fig.canvas.mpl_connect("button_press_event", on_click)

	plt.tight_layout()
	plt.show()
	plt.close("all")


def main():
	emb, labels, images = load_data()
	text_model = load_text_model()
	rows = []						# one table row per pair tried in this session

	while True:
		try:
			term1 = input("X-axis search term: ").strip()
			term2 = input("Y-axis search term: ").strip()
			top_n = int(input("Top N per term: "))
		except KeyboardInterrupt:
			break
		except ValueError:
			continue
		if not term1 or not term2 or top_n < 1:
			continue

		print("Encoding terms ...")
		t1, t2, ta, tb = encode_texts([term1, term2, IEAT_POLE_A, IEAT_POLE_B], text_model)
		x, y = emb @ t1, emb @ t2			# cosine similarity of every image to each term
		top_x, top_y = np.argsort(-x)[:top_n], np.argsort(-y)[:top_n]   # best first

		A, B, poles = pole_sets(labels, emb, ta, tb, top_n)
		m = {"term1": term1, "term2": term2, "top_n": top_n, "poles": poles,
			 **separability(x, y, t1, t2, top_x, top_y),
			 **ieat(emb, top_x, top_y, A, B)}

		# All pairs of this session as an aligned table; the new row is also saved.
		rows.append(m)
		save_row(m)
		table = pd.DataFrame(rows)[TABLE_COLUMNS]
		print("\n" + table.to_string(index=False, na_rep="", formatters=TERMINAL_FORMAT) + "\n")

		if SHOW_PLOT:
			make_plot(x, y, top_x, top_y, images, labels, term1, term2)


if __name__ == "__main__":
	main()