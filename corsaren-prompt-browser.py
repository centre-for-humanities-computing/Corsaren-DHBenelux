import sys, io, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import umap
from PIL import Image as PilImage
from sklearn.decomposition import PCA
from PySide6.QtWidgets import (
	QApplication, QMainWindow, QWidget, QSplitter, QScrollArea,
	QGridLayout, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
	QSizePolicy, QLineEdit, QPushButton,
)
from PySide6.QtCore import Qt, QThread, Signal, QPointF, QRectF
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# ── Data ── #
HF_REPO          = "chcaa/corsaren-caricatures"
IMAGE_COLUMN     = "image_bytes"
NAME_COLUMN      = "filename"
EMBEDDING_COLUMN = "eva02_clip"

TEXT_MODEL = ("EVA02-L-14", "merged2b_s4b_b131k")

# ── Map layout ── #
PCA_COMPONENTS      = 50        # PCA first reduces each embedding to 50 numbers so UMAP runs faster
RANDOM_SEED         = 42
UMAP_N_NEIGHBORS    = 15        # higher = more global structure, lower = more local clusters
UMAP_MIN_DIST       = 0.2       # lower = tighter clusters
UMAP_METRIC         = "cosine"

# ── Prompt colours ── #
PROMPT_COLORS = ["#41832d", "#96d17f", "#0082cb", "#b56340", "#e4ac45", "#d47a3f", "#8b5b1d", "#7d9871",
				 "#9999cc", "#457B9D", "#7DAAB1", "#9D7CB0", "#7e485e", "#7e7b4c", "#72BEC2"]

# ── Window ── #
WINDOW_SIZE    = (1600, 950)
SPLITTER_RATIO = [58, 42]

# ── Colours ── #
BG_COLOR        = "#ffffff"
PANEL_BG        = "#ffffff"
CARD_BG         = "#ffffff"
CARD_BG_QUERY   = "#fbeaea"
ACCENT_COLOR    = "#c0392b"     # selected dot, query card border and label
DOT_COLOR       = "#9a9aa8"     # normal dots, neighbour card borders
SELECTED_COLOR  = "#1e1e2e"
LINE_COLOR      = "#c8c8d0"
TEXT_COLOR      = "#222222"
TEXT_DIM        = "#6b6b6b"

# ── Thumbnails ── #
THUMB_SIZE = 130                   # after changing it, delete the thumbnail cache file
GRID_COLS  = 3                     # card columns in the panel

# =============================================================================
# FUNCTIONALITY
# =============================================================================

# ── Loading the data ── #

def load_data():
	"""Load filenames, embeddings and thumbnails.
	
	The file is downloaded once, then read from the local Hugging Face cache.
	"""
	from huggingface_hub import hf_hub_download
	print(f"Loading {HF_REPO} ...")
	file = hf_hub_download(HF_REPO, "corsaren-caricatures.parquet", repo_type="dataset")

	df = pd.read_parquet(file, columns=[NAME_COLUMN, EMBEDDING_COLUMN])
	names = df[NAME_COLUMN].tolist()
	emb = np.stack(df[EMBEDDING_COLUMN]).astype(np.float32)
	# Length 1, so a dot product is a cosine similarity.
	emb /= np.linalg.norm(emb, axis=1, keepdims=True)
	thumbs = load_thumbnails(file)
	print(f"  {len(names)} images")
	return names, emb, thumbs


def load_thumbnails(file):
	"""Small JPEG thumbnails for the whole corpus, cached in a file."""
	cache = BASE_DIR / f"thumbs_{HF_REPO.replace('/', '_')}.pkl"
	if cache.exists():
		with open(cache, "rb") as f:
			return pickle.load(f)

	print(f"Building thumbnail cache → {cache.name}")
	images = pd.read_parquet(file, columns=[IMAGE_COLUMN])[IMAGE_COLUMN]
	thumbs = []
	for i, image in enumerate(images):
		if i % 500 == 0:
			print(f"  {i}/{len(images)}")
		img = PilImage.open(io.BytesIO(image["bytes"])).convert("RGB")
		img.thumbnail((THUMB_SIZE * 2, THUMB_SIZE * 2))              # keeps aspect ratio
		buf = io.BytesIO()
		img.save(buf, "JPEG", quality=85)
		thumbs.append(buf.getvalue())
	with open(cache, "wb") as f:
		pickle.dump(thumbs, f)
	return thumbs


# ── Map layout and text model ── #

def compute_umap(emb):
	"""2D map position for every image."""
	print("PCA + UMAP ...")
	emb = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED).fit_transform(emb)
	return umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
					 metric=UMAP_METRIC, random_state=RANDOM_SEED).fit_transform(emb)


def load_text_model():
	"""The CLIP text encoder."""
	import torch, open_clip                 # imported here because they're slow to import
	device = "cuda" if torch.cuda.is_available() else "cpu"
	name, pretrained = TEXT_MODEL
	print(f"Loading text model {name} on {device} ...")
	model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
	model.visual = None                     # the image half isn't needed: frees memory
	model = model.to(device).eval()
	return model, open_clip.get_tokenizer(name), torch, device


# ── Prompt encoding ── #

def encode_prompt(text_model, prompt):
	"""A text prompt as a normalised CLIP embedding."""
	model, tokenizer, torch, device = text_model
	with torch.no_grad():                   # no training, so no gradients needed
		t = model.encode_text(tokenizer([prompt]).to(device))
	return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)[0]

# =============================================================================
# INTERFACE
# =============================================================================

# ── Small helpers ── #

def _label(text, color=TEXT_COLOR, size=10, bold=False, wrap=False):
	"""A centred text label with the given colour, font size and weight."""
	lbl = QLabel(text)
	lbl.setAlignment(Qt.AlignCenter)
	lbl.setWordWrap(wrap)
	lbl.setStyleSheet(f"color:{color}; font-size:{size}px; border:none;"
					  f"font-weight:{'bold' if bold else 'normal'};")
	return lbl


def _input_style(widget):
	"""Stylesheet for input boxes and buttons ("QLineEdit", "QSpinBox", ...)."""
	return (f"{widget}{{background:{CARD_BG};color:{TEXT_COLOR};"
			f"border:1px solid {LINE_COLOR};border-radius:4px;padding:3px 6px;font-size:12px;}}")


# ── Background thread ── #

class Worker(QThread):
	"""Runs a slow function in the background so the window stays responsive.
	"""
	done = Signal(object)

	def __init__(self, fn, *args):
		super().__init__()
		self.fn, self.args = fn, args

	def run(self):
		try:
			self.done.emit(self.fn(*self.args))
		except Exception as e:
			self.done.emit(e)


# ── Scatterplot ── #

class ScatterWidget(QWidget):
	"""Draws one dot per image and reports clicks and hovers.
	"""
	point_clicked = Signal(int)
	point_hovered = Signal(int)

	def __init__(self, emb2d):
		super().__init__()
		self.selected   = -1			# index of the highlighted dot, -1 = none
		self.hover_idx  = -1			# index of the dot under the mouse, -1 = none
		self.highlights = []			# prompt hits: list of (colour, indices)
		self.setMouseTracking(True)		# get mouse moves without a button pressed
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.setMinimumSize(400, 400)

		# Rescale the UMAP coordinates to 0–1 with a 5% margin
		m = 0.05
		xs, ys = emb2d[:, 0], emb2d[:, 1]
		self._nx = (xs - xs.min()) / np.ptp(xs) * (1 - 2 * m) + m
		self._ny = (ys - ys.min()) / np.ptp(ys) * (1 - 2 * m) + m

	def _pt(self, i):
		"""Screen position of dot i at the widget's current size."""
		return QPointF(self._nx[i] * self.width(), (1 - self._ny[i]) * self.height())

	def _nearest(self, pos):
		"""Index of the dot closest to a mouse position, or -1 if none is close."""
		dist = np.hypot(self._nx * self.width() - pos.x(),
						(1 - self._ny) * self.height() - pos.y())
		i = int(dist.argmin())
		return i if dist[i] < 17.5 else -1

	def select(self, idx):
		"""Highlight dot idx (called by the main window)."""
		self.selected = idx
		self.update()			# ask Qt to repaint soon

	def set_highlights(self, highlights):
		"""Set the coloured prompt hits and redraw the dots."""
		self.highlights = highlights
		self._render_base()
		self.update()

	def resizeEvent(self, e):
		"""Called by Qt when the widget changes size (always once before it is
		first shown): dot positions change, so redraw the dots."""
		self._render_base()

	def _render_base(self):
		"""Draw all dots and prompt hits into a cached picture."""
		dpr = self.devicePixelRatioF()      # >1 on high-resolution screens
		pm = QPixmap(int(self.width() * dpr), int(self.height() * dpr))
		pm.setDevicePixelRatio(dpr)
		pm.fill(QColor(BG_COLOR))
		p = QPainter(pm)
		p.setRenderHint(QPainter.Antialiasing)       # smooth edges

		# All dots, faded while prompts are active so the hits stand out.
		dot = QColor(DOT_COLOR)
		dot.setAlpha(90 if self.highlights else 170)
		p.setPen(Qt.NoPen)
		p.setBrush(dot)
		for i in range(len(self._nx)):
			p.drawEllipse(self._pt(i), 3, 3)

		# Prompt hits
		colors_of = {}
		for color, idxs in self.highlights:
			for i in idxs:
				colors_of.setdefault(int(i), []).append(color)

		# Draw each hit as a pie, split into equal slices if several prompts hit it
		for i, colors in sorted(colors_of.items(), key=lambda kv: len(kv[1])):
			r = 4.5 if len(colors) == 1 else 5      # pies a bit bigger, so the slices show
			c = self._pt(i)
			rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
			span = 5760 // len(colors)              # Qt measures angles in 1/16 degree: 5760 = full circle
			p.setPen(Qt.NoPen)
			for k, color in enumerate(colors):
				p.setBrush(QColor(color))
				p.drawPie(rect, 1440 - k * span, -span)
			p.setBrush(Qt.NoBrush)
			p.setPen(QPen(QColor(BG_COLOR), 0.8))   # thin outline in the background colour
			p.drawEllipse(c, r, r)
		p.end()
		self._base = pm

	def paintEvent(self, _):
		"""Called by Qt whenever the widget needs repainting."""
		p = QPainter(self)
		p.setRenderHint(QPainter.Antialiasing)
		p.drawPixmap(0, 0, self._base)
		# Ring around the dot under the mouse.
		if 0 <= self.hover_idx != self.selected:
			p.setBrush(Qt.NoBrush)
			p.setPen(QPen(QColor(TEXT_COLOR), 1.2))
			p.drawEllipse(self._pt(self.hover_idx), 6.5, 6.5)
		# Large dot in the accent colour for the selected image.
		if self.selected >= 0:
			p.setBrush(QColor(ACCENT_COLOR))
			p.setPen(QPen(QColor("white"), 1.5))
			p.drawEllipse(self._pt(self.selected), 6, 6)
		p.end()

	def mousePressEvent(self, e):
		if e.button() == Qt.LeftButton:
			idx = self._nearest(e.position())
			if idx >= 0:
				self.point_clicked.emit(idx)

	def mouseMoveEvent(self, e):
		idx = self._nearest(e.position())
		if idx != self.hover_idx:    # only repaint when the hovered dot changes
			self.hover_idx = idx
			self.update()
			self.point_hovered.emit(idx)


# ── Thumbnail cards and panel ── #

class ThumbnailCard(QWidget):
	"""One thumbnail with a small text line ("badge") and the filename."""
	clicked        = Signal(int)
	double_clicked = Signal(int)

	def __init__(self, idx, jpeg, name, badge, border, is_query=False):
		super().__init__()
		self.idx = idx
		bg = CARD_BG_QUERY if is_query else CARD_BG

		# Two looks for the card: normal, and selected (SELECTED_COLOR border).
		# QLabel{border:none} stops the labels inside inheriting the border.
		def style(px, col):
			return (f"QWidget{{background:{bg};border:{px}px solid {col};"
					f"border-radius:8px;}}"
					f"QLabel{{border:none;border-radius:0;}}")
		self._style_normal   = style(3 if is_query else 2, border)    # border width in pixels
		self._style_selected = style(3, SELECTED_COLOR)
		self.setStyleSheet(self._style_normal)
		self.setAttribute(Qt.WA_StyledBackground, True)   # needed to draw the border
		self.setCursor(Qt.PointingHandCursor)
		self.setFixedWidth(THUMB_SIZE + 16)               # 8 pixels padding on each side

		ly = QVBoxLayout(self)
		ly.setContentsMargins(8, 8, 8, 8)
		ly.setSpacing(3)

		# The image: decode the JPEG thumbnail and scale it to the card size.
		thumb = QLabel()
		thumb.setAlignment(Qt.AlignCenter)
		thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
		pm = QPixmap()
		pm.loadFromData(jpeg)
		dpr = self.devicePixelRatioF()
		pm = pm.scaled(int(THUMB_SIZE * dpr), int(THUMB_SIZE * dpr),
					   Qt.KeepAspectRatio, Qt.SmoothTransformation)
		pm.setDevicePixelRatio(dpr)
		thumb.setPixmap(pm)
		ly.addWidget(thumb)

		ly.addWidget(_label(badge, ACCENT_COLOR if is_query else TEXT_DIM,
							size=10 if is_query else 9, bold=is_query))
		ly.addWidget(_label(name, TEXT_COLOR, size=9, wrap=True))
		ly.addStretch()

	def set_selected(self, on):
		self.setStyleSheet(self._style_selected if on else self._style_normal)

	def mousePressEvent(self, e):
		if e.button() == Qt.LeftButton:
			self.clicked.emit(self.idx)

	def mouseDoubleClickEvent(self, e):
		# A double click also triggers mousePressEvent first, so the image
		# gets highlighted and then its similar images are shown.
		if e.button() == Qt.LeftButton:
			self.double_clicked.emit(self.idx)


class ImagePanel(QScrollArea):
	"""The scrollable grid of cards on the right."""
	card_clicked        = Signal(int)
	card_double_clicked = Signal(int)

	def __init__(self, names, thumbs):
		super().__init__()
		self.names, self.thumbs = names, thumbs
		self._cards = []
		self.setWidgetResizable(True)
		self.setStyleSheet(f"background:{PANEL_BG}; border:none;")

		# A scroll area scrolls one inner widget
		inner = QWidget()
		column = QVBoxLayout(inner)
		column.setContentsMargins(12, 12, 12, 12)
		self._title = _label("", TEXT_DIM, size=13, wrap=True)
		column.addWidget(self._title)
		self._grid = QGridLayout()
		self._grid.setSpacing(8)
		column.addLayout(self._grid)
		column.addStretch()
		self.setWidget(inner)

	def show_cards(self, title, title_color, items):
		"""Replace the title and all cards.

		items: list of (image index, badge text, border colour, is_query).
		"""
		self._title.setText(title)
		self._title.setStyleSheet(f"color:{title_color}; font-size:13px; font-weight:bold;")
		self.resize_cards([])                   # remove all cards
		self.resize_cards(items)
		self.verticalScrollBar().setValue(0)

	def resize_cards(self, items, selected=None):
		"""Grow or shrink the grid to `items`, keeping the cards it already has.
		When K or a prompt's N changes, only cards at the end are added or
		removed, which keeps the grid smooth."""
		keep = min(len(items), len(self._cards))
		for card in self._cards[keep:]:
			self._grid.removeWidget(card)
			card.hide()                         # disappear now; deleteLater() frees it soon
			card.deleteLater()
		del self._cards[keep:]
		for idx, badge, border, is_query in items[keep:]:
			card = ThumbnailCard(idx, self.thumbs[idx], self.names[idx], badge, border, is_query)
			card.clicked.connect(self.card_clicked)            # pass the signal on
			card.double_clicked.connect(self.card_double_clicked)
			if idx == selected:
				card.set_selected(True)
			self._grid.addWidget(card, *divmod(len(self._cards), GRID_COLS))  # row, column
			self._cards.append(card)

	def mark_selected(self, idx):
		"""Selected border on the card of image idx (if shown), normal on the rest."""
		for card in self._cards:
			card.set_selected(card.idx == idx)


# ── Main window ── #

class MainWindow(QMainWindow):
	"""Builds the interface and connects map, panel and prompts.

	Everything that selects an image (clicking a dot, clicking or
	double-clicking a card) goes through _select().
	"""

	def __init__(self, names, emb, thumbs):
		super().__init__()
		self.names  = names
		self.emb    = emb
		self.n      = len(names)
		self._scatter    = None       # created once the map is computed
		self._text_model = None       # set once the text model is loaded
		self._queries    = []         # active prompts
		self._selected   = None       # the highlighted image
		self._neighbours_of = None    # image whose neighbours the panel shows, if any
		self._hits_of       = None    # prompt whose hits the panel shows, if any

		self.resize(*WINDOW_SIZE)
		self.setStyleSheet(f"QMainWindow{{background:{BG_COLOR};}}")
		self._build_ui(thumbs)

		# Start the two slow jobs in the background.
		self._umap = Worker(compute_umap, emb)
		self._umap.done.connect(self._on_umap_done)
		self._umap.start()
		self._loader = Worker(load_text_model)
		self._loader.done.connect(self._on_text_model_loaded)
		self._loader.start()

	# ── Building the interface ── #

	def _bar(self):
		"""An empty horizontal bar (used for the two bars at the top)."""
		bar = QWidget()
		bar.setObjectName("bar")
		bar.setAttribute(Qt.WA_StyledBackground, True)
		bar.setStyleSheet(f"#bar{{background:{PANEL_BG}; border-bottom:1px solid {LINE_COLOR};}}")
		bar.setFixedHeight(46)
		row = QHBoxLayout(bar)
		row.setContentsMargins(16, 0, 16, 0)
		row.setSpacing(10)
		return bar, row

	def _build_ui(self, thumbs):
		# ── Top bar: image count, K ──
		top, row = self._bar()
		row.addWidget(_label(f"{self.n} images", TEXT_COLOR, size=13, bold=True))
		row.addStretch()                        # pushes the rest to the right
		row.addWidget(_label("K =", TEXT_COLOR, size=12))
		self._k_spin = QSpinBox()               # number of similar images shown
		self._k_spin.setRange(1, self.n - 1)      # all other images
		self._k_spin.setValue(9)
		self._k_spin.setFixedWidth(58)
		self._k_spin.setStyleSheet(_input_style("QSpinBox"))
		self._k_spin.valueChanged.connect(self._on_k_changed)
		row.addWidget(self._k_spin)

		# ── Prompt bar: prompt box, prompt chips, clear ──
		pbar, prow = self._bar()
		self._prompt_box = QLineEdit()
		self._prompt_box.setFixedWidth(300)
		self._prompt_box.setStyleSheet(_input_style("QLineEdit"))
		self._prompt_box.setPlaceholderText("Loading text model…")
		self._prompt_box.setEnabled(False)      # until the text model is loaded
		self._prompt_box.returnPressed.connect(self._add_prompt)
		prow.addWidget(self._prompt_box)

		# The chips sit in a horizontally scrolling strip, in case there are many.
		chip_area = QScrollArea()
		chip_area.setWidgetResizable(True)
		chip_area.setFixedHeight(40)
		chip_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		chip_area.setStyleSheet(f"background:{PANEL_BG}; border:none;")
		chip_inner = QWidget()
		self._chips = QHBoxLayout(chip_inner)
		self._chips.setContentsMargins(0, 0, 0, 0)
		self._chips.setSpacing(6)
		self._chips.addStretch()
		chip_area.setWidget(chip_inner)
		prow.addWidget(chip_area, stretch=1)
		clear = QPushButton("Clear")
		clear.setStyleSheet(_input_style("QPushButton"))
		clear.clicked.connect(lambda: self._remove_prompts(self._queries))
		prow.addWidget(clear)

		# ── Map | panel, with a draggable divider ──
		self._splitter = QSplitter(Qt.Horizontal)
		self._splitter.setHandleWidth(3)
		self._splitter.setStyleSheet(f"QSplitter::handle{{background:{LINE_COLOR};}}")
		self._placeholder = _label("Computing UMAP…", TEXT_DIM, size=14)
		self._splitter.addWidget(self._placeholder)
		self._panel = ImagePanel(self.names, thumbs)
		self._panel.card_clicked.connect(lambda i: self._select(i, neighbours=False))
		self._panel.card_double_clicked.connect(lambda i: self._select(i, neighbours=True))
		self._splitter.addWidget(self._panel)
		self._splitter.setSizes([int(WINDOW_SIZE[0] * r / 100) for r in SPLITTER_RATIO])

		self.statusBar().setStyleSheet(
			f"color:{TEXT_DIM};font-size:11px;background:{PANEL_BG};")
		self.statusBar().showMessage("Loading…")

		# Stack the two bars and the splitter vertically.
		central = QWidget()
		ly = QVBoxLayout(central)
		ly.setContentsMargins(0, 0, 0, 0)
		ly.setSpacing(0)
		ly.addWidget(top)
		ly.addWidget(pbar)
		ly.addWidget(self._splitter)
		self.setCentralWidget(central)

	# ── Results from the background workers ── #

	def _on_umap_done(self, emb2d):
		"""The map is computed: swap the placeholder for the real map."""
		if isinstance(emb2d, Exception):
			self._placeholder.setText(f"UMAP failed:\n{emb2d}")
			print(f"UMAP error: {emb2d}")
			return
		self._scatter = ScatterWidget(emb2d)
		self._scatter.point_clicked.connect(self._on_map_click)
		self._scatter.point_hovered.connect(self._on_hover)
		self._splitter.replaceWidget(0, self._scatter)
		self._placeholder.deleteLater()
		self._splitter.setSizes([int(self.width() * r / 100) for r in SPLITTER_RATIO])
		self._refresh_hits()                    # colour prompts added before the map was ready
		if self._selected is not None:          # and an image selected before it was ready
			self._scatter.select(self._selected)
		self.statusBar().showMessage("Click a point to see similar images")

	def _on_text_model_loaded(self, result):
		"""The text model is loaded (or failed): enable the prompt box."""
		if isinstance(result, Exception):
			self._prompt_box.setPlaceholderText("Text model failed to load")
			self.statusBar().showMessage(f"Text model error: {result}")
			print(f"Text model error: {result}")
			return
		self._text_model = result
		self._prompt_box.setEnabled(True)
		self._prompt_box.setPlaceholderText("Search prompt")

	# ── Selecting images ── #

	def _on_map_click(self, idx):
		"""Click a dot to select it; click the selected dot again to deselect."""
		if idx == self._selected:
			self._selected = None
			self._scatter.select(-1)              # -1 = no selected dot
			self._panel.mark_selected(None)       # no selected card border
			self.statusBar().clearMessage()
		else:
			self._select(idx, neighbours=True)

	def _select(self, idx, neighbours):
		"""Highlight image idx on the map; with neighbours=True, also show
		its most similar images in the panel."""
		self._selected = idx
		if self._scatter is not None:
			self._scatter.select(idx)
		if neighbours:                          # its card is the query card
			self._panel.show_cards("Most similar images", TEXT_DIM, self._neighbour_items(idx))
			self._neighbours_of, self._hits_of = idx, None
		else:
			self._panel.mark_selected(idx)      # selected border on its card
		self.statusBar().showMessage(self._describe(idx))

	def _neighbour_items(self, idx):
		"""Image idx as the query card, followed by its K most similar images."""
		sims = self.emb @ self.emb[idx]         # cosine similarity, embeddings are normalised
		sims[idx] = -np.inf                     # not itself
		best = np.argsort(-sims)[:self._k_spin.value()]
		return [(idx, "QUERY", ACCENT_COLOR, True)] + \
			   [(i, f"s = {sims[i]:.3f}", DOT_COLOR, False) for i in best]

	def _on_k_changed(self, _):
		"""K changed: grow or shrink the neighbours if they're currently shown."""
		if self._neighbours_of is not None:
			self._panel.resize_cards(self._neighbour_items(self._neighbours_of), self._selected)

	def _describe(self, idx):
		"""Status bar text: filename, then score and rank for each prompt."""
		parts = [self.names[idx]]
		for q in self._queries:
			s = q["scores"][idx]
			rank = int((q["scores"] > s).sum()) + 1        # 1 + images that score higher
			parts.append(f"{q['prompt']}: {s:.3f} (#{rank})")
		return "   ·   ".join(parts)
	
	def _on_hover(self, idx):
		self.statusBar().showMessage(self._describe(idx) if idx >= 0 else "")

	# ── Text prompts ── #

	def _add_prompt(self):
		"""Encode the typed prompt, score every image, add a chip."""
		prompt = self._prompt_box.text().strip()
		if not prompt:
			return
		if any(q["prompt"] == prompt for q in self._queries):
			self.statusBar().showMessage(f"“{prompt}” is already active.")
			return

		scores = self.emb @ encode_prompt(self._text_model, prompt)   # cosine similarity
		used = {q["color"] for q in self._queries}
		color = next((c for c in PROMPT_COLORS if c not in used),        # first free colour
					 PROMPT_COLORS[len(self._queries) % len(PROMPT_COLORS)])

		q = {"prompt": prompt, "color": color, "scores": scores, "n": 50}
		self._chips.insertWidget(self._chips.count() - 1, self._make_chip(q))
		self._queries.append(q)
		self._prompt_box.clear()
		self._refresh_hits()
		self._show_prompt_hits(q)

	def _make_chip(self, q):
		"""The chip for one prompt: '● prompt' | top-N box | '×'."""
		chip = QWidget()
		h = QHBoxLayout(chip)
		h.setContentsMargins(0, 0, 0, 0)
		h.setSpacing(0)
		base = (f"background:{CARD_BG};color:{q['color']};border:1px solid {q['color']};"
				f"padding:3px 8px;font-size:11px;")

		name = QPushButton(f"● {q['prompt']}")
		name.setStyleSheet(f"QPushButton{{{base}border-top-left-radius:10px;"
						   f"border-bottom-left-radius:10px;border-right:none;}}")
		name.clicked.connect(lambda: self._show_prompt_hits(q))

		n_box = QSpinBox()
		n_box.setRange(1, self.n)
		n_box.setValue(q["n"])
		n_box.setToolTip("Top N: how many of this prompt's best matches are shown")
		n_box.setStyleSheet(f"QSpinBox{{{base}padding:0px 2px;border-right:none;}}")
		n_box.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
		n_box.setAlignment(Qt.AlignCenter)
		n_box.setFixedSize(42, name.sizeHint().height())
		n_box.valueChanged.connect(lambda n: self._set_n(q, n))

		x = QPushButton("×")
		x.setStyleSheet(f"QPushButton{{{base}border-top-right-radius:10px;"
						f"border-bottom-right-radius:10px;}}")
		x.clicked.connect(lambda: self._remove_prompts([q]))

		for w in (name, n_box, x):
			h.addWidget(w)
		q["chip"] = chip
		return chip

	def _set_n(self, q, n):
		"""A prompt's top N changed: update the map, and its grid if shown."""
		q["n"] = n
		self._refresh_hits()
		if q is self._hits_of:
			self._show_prompt_hits(q, resize=True)

	def _remove_prompts(self, queries):
		for q in list(queries):                 # copy: we remove while looping
			self._queries.remove(q)
			self._chips.removeWidget(q["chip"])
			q["chip"].hide()
			q["chip"].deleteLater()
			if q is self._hits_of:
				self._hits_of = None            # its hits stay visible until replaced
		self._refresh_hits()

	def _refresh_hits(self):
		"""Recompute every prompt's hits from its own top N; update map and overlap."""
		for q in self._queries:
			q["hits"] = np.argsort(-q["scores"])[:q["n"]]   # best first
		if self._scatter is not None:
			self._scatter.set_highlights([(q["color"], q["hits"]) for q in self._queries])
		if len(self._queries) > 1:
			# images that are a hit for more than one prompt
			counts = np.bincount(np.concatenate([q["hits"] for q in self._queries]),
								 minlength=self.n)
			shared = int((counts > 1).sum())
			self.statusBar().showMessage(f"{shared} images are hits for more than one prompt.")
		else:
			self.statusBar().clearMessage()     # no overlap to report

	def _show_prompt_hits(self, q, resize=False):
		"""Show a prompt's hits as cards, best first with rank and score.
		resize=True: only grow or shrink the cards already shown (N changed)."""
		items = [(i, f"#{r + 1}   s = {q['scores'][i]:.3f}", q["color"], False)
				 for r, i in enumerate(q["hits"])]
		if resize:
			self._panel.resize_cards(items, self._selected)
		else:
			self._panel.show_cards(f"“{q['prompt']}”", q["color"], items)
			self._neighbours_of, self._hits_of = None, q

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
	names, emb, thumbs = load_data()
	app = QApplication(sys.argv)
	app.setStyle("Fusion")
	window = MainWindow(names, emb, thumbs)
	window.show()
	sys.exit(app.exec())                         # runs until the window is closed


if __name__ == "__main__":
	main()