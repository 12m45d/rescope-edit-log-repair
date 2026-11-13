"""The two encoders of the retrieval reference (PROTOCOL.md, section 6, "Review round 1", item R4) and the storage form of an embedding.

IMAGES     ImageEncoder: the CLIP vision tower that is part of the local LLaVA-1.5 checkpoint (models/llava-1.5-7b).  ONLY the tower is
           loaded: the 391 tensors `vision_tower.vision_model.*` are read from the checkpoint's safetensors shard (safetensors lazy
           access; the language model and the projector are never read), into a transformers CLIPVisionModel built from the
           checkpoint's `vision_config` (ViT-L/14, 336 px, 24 layers, width 1024, quick_gelu).  Weights: the checkpoint's float16
           values cast (exactly) to float32; forward in float32, eval mode, one image per forward (B = 1: the embedding of an image
           does not depend on which other images are embedded with it).
           Preprocessing = the checkpoint's own image processor (AutoImageProcessor: CLIPImageProcessor of preprocessor_config.json,
           the processor LLaVA receives in this project): PIL decode, convert("RGB") (no EXIF rotation, as the project's loader),
           resize of the shortest edge to 336 (bicubic), centre crop 336 x 336, rescale 1 / 255, normalise with the CLIP mean / std.
           Embedding = POOLED OUTPUT of the tower = post_layernorm(CLS token of the last layer) (`pooler_output`, 1024 numbers; the
           checkpoint has no visual projection).  The same encoder serves both backbones.
QUESTIONS  QuestionEncoder: the local sentence encoder models/all-MiniLM-L6-v2 through sentence-transformers (installed: 6.1.0), i.e.
           the model card's recipe: BERT tokenizer (uncased, max 256 tokens), mean pooling of the token embeddings over the
           attention mask, L2 normalisation; float32, CPU, ONE string per forward (no padding: the embedding of a string does not
           depend on the batch).  Input = the question key = question.strip().lower().  backend="transformers" is the model
           card's second recipe (AutoTokenizer + AutoModel + mean pooling + normalisation), used as an independent check.
STORAGE    finalize(x): x -> x / |x| in float64 -> every component rounded to the nearest multiple of 2^-24 -> float32.
           The stored vectors are float32 and unit-norm (| |e| - 1 | < 1e-7; every component within 3e-8 of the normalised value).
           Reason for the grid: for two such vectors every product e_k f_k is a multiple of 2^-48 and every partial sum of products
           is at most (|e| |f|) in magnitude (Cauchy-Schwarz), so it is exactly representable in float64: the float64 dot product
           is EXACT, whatever the BLAS, the thread count, the batch shape or the summation order.  Scores (retr.py) are therefore
           bit-for-bit reproducible functions of the stored files.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import rcommon as C  # noqa: E402

GRID = float(2 ** 24)
IMG_DIM, Q_DIM = 1024, 384
TOWER_PREFIX = "vision_tower."
Q_MAX_TOKENS = 256


def finalize(x) -> np.ndarray:
    """(n, d) any float -> unit-norm float32 on the 2^-24 grid (module docstring).  A row with a non-finite or zero norm raises."""
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    nrm = np.sqrt((x * x).sum(1, keepdims=True))
    if not bool(np.isfinite(nrm).all()) or bool((nrm <= 0).any()):
        raise ValueError("embedding with a zero or non-finite norm")
    q = np.rint(x / nrm * GRID) / GRID
    out = q.astype(np.float32)
    assert bool((out.astype(np.float64) == q).all())                    # multiples of 2^-24 up to 1 are float32 numbers
    return out


def on_grid(e) -> bool:
    """Every component of a stored matrix is a multiple of 2^-24 (the condition of the exact dot product)."""
    v = np.asarray(e, dtype=np.float64) * GRID
    return bool((v == np.rint(v)).all())


def load_image(path: str):
    """PIL RGB image of a file (the project's loader, vendor_v7.vlm.load_image: no EXIF transpose; very large images allowed)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return im.convert("RGB")


def _versions() -> dict:
    import PIL
    import torch
    import transformers
    v = {"torch": torch.__version__, "transformers": transformers.__version__, "PIL": PIL.__version__, "numpy": np.__version__}
    for m in ("torchvision", "safetensors", "sentence_transformers"):
        try:
            v[m] = __import__(m).__version__
        except Exception:                                               # noqa: BLE001
            v[m] = None
    return v


class ImageEncoder:
    """enc = ImageEncoder(device); enc.embed_paths(paths) -> (n, 1024) float32 (finalize); enc.info = what was loaded and how."""

    def __init__(self, device: str = "cuda:0"):
        import torch
        from safetensors import safe_open
        from transformers import AutoImageProcessor, CLIPVisionConfig, CLIPVisionModel
        self.torch, self.device = torch, torch.device(device)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, False
        cfg = C.jload(os.path.join(C.LLAVA_DIR, "config.json"))
        vcfg = {k: v for k, v in cfg["vision_config"].items() if k not in ("model_type", "vocab_size")}
        vc = CLIPVisionConfig(**vcfg)
        wm = C.jload(os.path.join(C.LLAVA_DIR, "model.safetensors.index.json"))["weight_map"]
        keys = sorted(k for k in wm if k.startswith(TOWER_PREFIX))
        model = CLIPVisionModel(vc)
        want = list(model.state_dict().keys())
        pref = TOWER_PREFIX if want[0].startswith("vision_model.") else TOWER_PREFIX + "vision_model."
        sd, files, src_dtypes = {}, sorted({wm[k] for k in keys}), set()
        for fn in files:                                                # lazy access: only the tower's tensors are read
            with safe_open(os.path.join(C.LLAVA_DIR, fn), framework="pt", device="cpu") as f:
                for k in keys:
                    if wm[k] == fn:
                        t = f.get_tensor(k)
                        src_dtypes.add(str(t.dtype))
                        sd[k[len(pref):]] = t.to(torch.float32)
        if set(sd) != set(want):
            raise RuntimeError(f"the vision tower cannot be loaded alone: missing {sorted(set(want) - set(sd))[:5]}, unexpected {sorted(set(sd) - set(want))[:5]}")
        model.load_state_dict(sd, strict=True)
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.proc = AutoImageProcessor.from_pretrained(C.LLAVA_DIR)
        h = __import__("hashlib").sha1()
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].numpy().tobytes())
        self.info = {"encoder": "CLIP vision tower of the local LLaVA-1.5 checkpoint, loaded alone", "checkpoint": C.LLAVA_DIR,
                     "tensors": f"{len(keys)} tensors with the prefix '{pref}' from {files} (safetensors lazy access; nothing else is read)",
                     "n_parameters": int(sum(v.numel() for v in sd.values())), "weights_sha1": h.hexdigest(),
                     "checkpoint_dtype": sorted(src_dtypes), "forward_dtype": "float32 (the checkpoint's float16 values, cast exactly)",
                     "model_class": type(self.model).__name__, "vision_config": vcfg,
                     "config_defaults_used": {"hidden_act": vc.hidden_act, "layer_norm_eps": vc.layer_norm_eps, "num_channels": vc.num_channels},
                     "embedding": "pooler_output = post_layernorm(CLS token of the last layer), 1024-d; no visual projection (not in the checkpoint)",
                     "batch": "one image per forward", "image_processor_class": f"{type(self.proc).__module__}.{type(self.proc).__name__}",
                     "image_processor": json.loads(json.dumps(self.proc.to_dict(), default=lambda o: list(o) if isinstance(o, tuple) else str(o))),
                     "decode": "PIL.Image.open(path).convert('RGB'), no EXIF transpose, Image.MAX_IMAGE_PIXELS = None (as vendor_v7.vlm.load_image)",
                     "preprocessing": "convert RGB -> resize shortest edge 336 (bicubic, resample 3) -> centre crop 336 x 336 -> rescale 1/255 -> "
                                      "normalise (CLIP mean / std) -> float32 (1, 3, 336, 336)",
                     "storage": "finalize: L2 normalisation in float64, components rounded to multiples of 2^-24, float32",
                     "device": str(self.device), "tf32": False, "cudnn_deterministic": True, "versions": _versions()}

    def pixel_values(self, path: str):
        """(1, 3, 336, 336) float32 CPU tensor of an image file (the checkpoint's image processor)."""
        return self.proc(images=[load_image(path)], return_tensors="pt")["pixel_values"].to(self.torch.float32)

    def raw(self, pixel_values):
        """Pooled output (1, 1024) float32 of one preprocessed image."""
        with self.torch.no_grad():
            out = self.model(pixel_values=pixel_values.to(self.device))
        return out.pooler_output.float().cpu()

    def embed_paths(self, paths, threads: int = 4, progress=None) -> np.ndarray:
        """(n, 1024) float32 embeddings (finalize) of image files, in order; decoding and preprocessing in `threads` threads,
        one forward per image."""
        from concurrent.futures import ThreadPoolExecutor
        out = np.zeros((len(paths), IMG_DIM), dtype=np.float32)
        with ThreadPoolExecutor(max(1, int(threads))) as ex:
            for i, pv in enumerate(ex.map(self.pixel_values, paths)):
                out[i] = finalize(self.raw(pv).numpy())[0]
                if progress is not None:
                    progress(i + 1)
        return out


class QuestionEncoder:
    """enc = QuestionEncoder(backend="sentence-transformers" | "transformers"); enc.embed(strings) -> (n, 384) float32 (finalize)."""

    def __init__(self, backend: str = "sentence-transformers", device: str = "cpu"):
        import torch
        self.torch, self.backend, self.device = torch, backend, device
        if backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            self.st = SentenceTransformer(C.MINILM_DIR, device=device)
            self.st.eval()
            self.max_tokens = int(self.st.max_seq_length)
            mods = [type(m).__name__ for m in self.st]
        elif backend == "transformers":
            from transformers import AutoModel, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(C.MINILM_DIR)
            self.model = AutoModel.from_pretrained(C.MINILM_DIR).eval().requires_grad_(False).to(device)
            self.max_tokens = int(C.jload(os.path.join(C.MINILM_DIR, "sentence_bert_config.json"))["max_seq_length"])
            mods = ["AutoTokenizer", "AutoModel", "mean pooling over the attention mask", "L2 normalisation"]
        else:
            raise ValueError("backend: 'sentence-transformers' | 'transformers'")
        if self.max_tokens != Q_MAX_TOKENS:
            raise RuntimeError(f"max_seq_length {self.max_tokens}, expected {Q_MAX_TOKENS}")
        self.info = {"encoder": "all-MiniLM-L6-v2 (local)", "checkpoint": C.MINILM_DIR, "backend": backend, "modules": mods,
                     "recipe": "model card: tokenizer (BERT, uncased) -> token embeddings -> mean pooling over the attention mask -> L2 normalisation",
                     "max_seq_length": self.max_tokens, "input": "the question key = question.strip().lower()", "batch": "one string per forward (no padding)",
                     "forward_dtype": "float32", "device": device,
                     "storage": "finalize: L2 normalisation in float64, components rounded to multiples of 2^-24, float32", "versions": _versions()}

    def n_tokens(self, s: str) -> int:
        tok = self.st.tokenizer if self.backend == "sentence-transformers" else self.tok
        return len(tok(s, add_special_tokens=True, truncation=False)["input_ids"])

    def raw(self, s: str) -> np.ndarray:
        """(384,) float32 normalised embedding of one string, as the backend returns it."""
        torch = self.torch
        if self.backend == "sentence-transformers":
            with torch.no_grad():
                e = self.st.encode([s], batch_size=1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(e[0], dtype=np.float32)
        enc = self.tok([s], padding=True, truncation=True, max_length=self.max_tokens, return_tensors="pt").to(self.device)
        with torch.no_grad():
            tokemb = self.model(**enc)[0]                               # model card: "First element of model_output contains all token embeddings"
        mask = enc["attention_mask"].unsqueeze(-1).expand(tokemb.size()).float()
        pooled = torch.sum(tokemb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)[0].float().cpu().numpy()

    def embed(self, strings, progress=None) -> np.ndarray:
        out = np.zeros((len(strings), Q_DIM), dtype=np.float32)
        for i, s in enumerate(strings):
            out[i] = finalize(self.raw(s))[0]
            if progress is not None:
                progress(i + 1)
        return out
