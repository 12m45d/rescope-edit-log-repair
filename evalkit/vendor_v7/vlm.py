"""v7 VLM wrapper: one batched, right-padded, teacher-forced interface for
BLIP2-OPT-2.7B, LLaVA-1.5-7B and Qwen2.5-VL-3B (HF checkpoints, transformers 5.17).

Conventions (see [redacted: related project] for the full list):

* All parameters are frozen (``requires_grad_(False)``); default dtypes are
  fp16 (BLIP2, LLaVA) and bf16 (Qwen2.5-VL, its native precision).
* Token ids are built by hand from three pieces, so that image-token expansion
  and the prompt/answer boundary are exact:
      prefix ids (BOS / image placeholders)  +  tok(prompt text)  +  tok(answer)
  ``answer_ids`` are tokenised on their own and are tested to equal the tail of
  the joint tokenisation of ``prompt text + sep + answer`` (tests/test_tokenization.py).
* Formats (``fmt``):
    ``vqa``   BLIP2/LLaVA: ``"Question: {q} Short answer:"`` + ``" " + answer``
              (VLKEB VQADataset template).  BLIP2 sequence = [32 query slots][</s>][text];
              LLaVA sequence = [<s>][576 image slots][sep + text] with sep="\\n" by
              default (``llava_image_sep``; "" reproduces VLKEB's wrapper exactly).
              Qwen: chat template, user = [image, question], assistant = answer.
    ``plain`` BLIP2/LLaVA: raw ``prompt + " " + answer`` (VLKEB T-Loc); Qwen: chat
              template without image (same as ``vqa`` without image).
    [not released: code that no released code calls; it served the related project (README, "What can be reproduced")]
* Positions: ``last_prompt_idx[i]`` is the last prompt token of row i (the token
  whose logits predict the first answer token); ``pred_pos[i, j] =
  answer_start[i] - 1 + j`` are the positions whose logits predict answer token j.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import contextlib
import os
from typing import Iterable, Sequence

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
Image.MAX_IMAGE_PIXELS = None  # trusted local data; some VLKEB images exceed 89 Mpx

ROOT = f"{_RP.EXT_VIEW}"
MODEL_DIRS = {
    "blip2": os.path.join(ROOT, "data", "models", "blip2-opt-2.7b"),
    "llava": os.path.join(ROOT, "data", "models", "llava-1.5-7b"),
    "qwen25vl": os.path.join(ROOT, "data", "models", "Qwen2.5-VL-3B"),
}
VLKEB_IMG_DIR = os.path.join(ROOT, "data", "datasets", "VLKEB", "mmkb_images")
DEFAULT_DTYPE = {"blip2": torch.float16, "llava": torch.float16, "qwen25vl": torch.bfloat16}
QA_TEMPLATE = "Question: {} Short answer:"
QWEN_MIN_PIXELS = 256 * 28 * 28   # [redacted: related project] (256..1024 visual tokens)
QWEN_MAX_PIXELS = 1024 * 28 * 28
FORMATS = ("vqa", "plain")   # [not released: code that no released code calls; it served the related project (README, "What can be reproduced")]
_Q_SENTINEL = "<<<V7_QUESTION>>>"


class _StopForward(Exception):
    """Raised inside a layer hook to truncate the forward pass."""


def load_image(img, img_dir: str = VLKEB_IMG_DIR) -> Image.Image:
    """PIL image from a PIL image or a path (relative paths are resolved under ``img_dir``)."""
    if isinstance(img, Image.Image):
        return img if img.mode == "RGB" else img.convert("RGB")
    img = _RP.expand(img)                                # released records hold ${RESCOPE_EXT_VIEW}/... for images outside the default benchmark
    path = img if os.path.isabs(img) else os.path.join(img_dir, img)
    with Image.open(path) as im:
        return im.convert("RGB")


class Capture:
    """Container filled by :meth:`VLM.capture`: ``inp[l]`` / ``out[l]`` tensors per layer."""

    def __init__(self):
        self.inp: dict[int, torch.Tensor] = {}
        self.out: dict[int, torch.Tensor] = {}
        self.rows: torch.Tensor | None = None  # row index of each vector for flattened selections

    def __repr__(self):
        f = lambda d: {k: tuple(v.shape) for k, v in d.items()}
        return f"Capture(inp={f(self.inp)}, out={f(self.out)})"


class VLM:
    """Frozen multimodal LM with batched teacher-forced inputs.

    Parameters
    ----------
    backbone : "blip2" | "llava" | "qwen25vl"
    device   : torch device string (we only use cuda:0 in v7)
    dtype    : None -> DEFAULT_DTYPE[backbone]
    llava_image_sep : text between LLaVA's image slots and the prompt ("\\n" default, "" = VLKEB)
    load_model : False builds tokenizer + image processor only (for CPU tokenisation tests)
    """

    def __init__(self, backbone: str, device: str = "cuda:0", dtype: torch.dtype | None = None,
                 llava_image_sep: str = "\n", load_model: bool = True,
                 attn_implementation: str = "sdpa"):
        if backbone not in MODEL_DIRS:
            raise ValueError(f"unknown backbone {backbone!r}; choose from {list(MODEL_DIRS)}")
        from transformers import AutoProcessor, AutoTokenizer
        self.backbone = backbone
        self.device = torch.device(device)
        self.dtype = dtype or DEFAULT_DTYPE[backbone]
        self.llava_image_sep = llava_image_sep
        path = MODEL_DIRS[backbone]
        self.tok = AutoTokenizer.from_pretrained(path)
        proc = AutoProcessor.from_pretrained(path)
        self.image_processor = proc.image_processor
        if backbone == "qwen25vl":
            self.image_processor.min_pixels = QWEN_MIN_PIXELS
            self.image_processor.max_pixels = QWEN_MAX_PIXELS
            self._qwen_tmpl = {
                True: proc.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _Q_SENTINEL}]}],
                    tokenize=False, add_generation_prompt=True),
                False: proc.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": _Q_SENTINEL}]}],
                    tokenize=False, add_generation_prompt=True),
            }
        del proc
        self.bos_id = self.tok.bos_token_id if backbone in ("blip2", "llava") else None
        self.pad_id = self.tok.pad_token_id
        self.m = None
        if load_model:
            self._load_model(path, attn_implementation)
        self._init_constants()

    # ------------------------------------------------------------------ setup
    def _load_model(self, path, attn_implementation):
        from transformers import (Blip2ForConditionalGeneration, LlavaForConditionalGeneration,
                                  Qwen2_5_VLForConditionalGeneration)
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        cls = {"blip2": Blip2ForConditionalGeneration, "llava": LlavaForConditionalGeneration,
               "qwen25vl": Qwen2_5_VLForConditionalGeneration}[self.backbone]
        self.m = cls.from_pretrained(path, dtype=self.dtype, device_map={"": str(self.device)},
                                     attn_implementation=attn_implementation)
        self.m.eval()
        self.m.requires_grad_(False)
        assert not any(p.requires_grad for p in self.m.parameters())

    def _init_constants(self):
        import json
        cfg = json.load(open(os.path.join(MODEL_DIRS[self.backbone], "config.json")))
        if self.backbone == "blip2":
            self.image_token_id = cfg["image_token_index"]            # 50265
            self.n_layers = cfg["text_config"]["num_hidden_layers"]   # 32
            self.n_query = cfg["num_query_tokens"]                    # 32
        elif self.backbone == "llava":
            self.image_token_id = cfg["image_token_index"]            # 32000
            self.n_layers = 32
            v = cfg["vision_config"]
            n = (v["image_size"] // v["patch_size"]) ** 2
            self.n_llava_img = n + (1 if cfg.get("vision_feature_select_strategy") == "full" else 0)  # 576
        else:
            self.image_token_id = cfg["image_token_id"]               # 151655
            self.n_layers = cfg["num_hidden_layers"]                  # 36
            self.merge = cfg["vision_config"]["spatial_merge_size"]   # 2
        if self.m is not None:
            mod = self.edit_module(0)
            self.d_in, self.d_out = mod.in_features, mod.out_features
            assert len(self._layers()) == self.n_layers

    # ------------------------------------------------------------- modules
    def _layers(self):
        if self.backbone == "blip2":
            return self.m.language_model.model.decoder.layers
        return self.m.model.language_model.layers

    def edit_module_name(self, l: int) -> str:
        if self.backbone == "blip2":
            return f"language_model.model.decoder.layers.{l}.fc2"
        return f"model.language_model.layers.{l}.mlp.down_proj"

    def layer_module_name(self, l: int) -> str:
        if self.backbone == "blip2":
            return f"language_model.model.decoder.layers.{l}"
        return f"model.language_model.layers.{l}"

    def edit_module(self, l: int) -> torch.nn.Linear:
        mod = self.m.get_submodule(self.edit_module_name(l))
        assert isinstance(mod, torch.nn.Linear), type(mod)
        return mod

    def layer_module(self, l: int) -> torch.nn.Module:
        """Decoder block l; its output is the residual stream after layer l."""
        return self.m.get_submodule(self.layer_module_name(l))

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.m.get_output_embeddings()

    # -------------------------------------------------------- tokenisation
    def _tok(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"] if text else []

    def text_parts(self, prompt: str, fmt: str, with_image: bool):
        """(prefix_ids, prompt_text, answer_sep) for one row.

        prefix_ids are id-level pieces placed before tok(prompt_text): BOS and image
        placeholders (Qwen's single <|image_pad|> lives inside prompt_text and is
        expanded at id level later).
        """
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}")
        b = self.backbone
        # [not released: code that no released code calls; it served the related project (README, "What can be reproduced")]
        if b == "qwen25vl":   # vqa and plain both use the chat template
            return [], self._qwen_tmpl[with_image].replace(_Q_SENTINEL, prompt), ""
        # BLIP2/LLaVA follow VLKEB's VQADataset template; Qwen's prompt is the raw question
        # inside its chat template (VLKEB never ran Qwen, so there is no VLKEB convention to copy).
        text = prompt if b == "qwen25vl" else (QA_TEMPLATE.format(prompt) if fmt == "vqa" else prompt)
        if b == "blip2":
            prefix = ([self.image_token_id] * self.n_query if with_image else []) + [self.bos_id]
            return prefix, text, " "
        # llava: [<s>][576 x <image>][sep + text]  (tokenised as one chunk, like LLaVA's
        # tokenizer_image_token / VLKEB's wrapper)
        if with_image:
            return [self.bos_id] + [self.image_token_id] * self.n_llava_img, self.llava_image_sep + text, " "
        return [self.bos_id], text, " "

    def answer_and_joint(self, text: str, answer: str, sep: str = " ") -> tuple[list[int], list[int]]:
        """(answer ids, joint ids) where ``text`` is the prompt text of :meth:`text_parts`.

        The scored answer is the longest suffix of the joint tokenisation ``text + sep + answer``
        that the answer's own tokenisation also ends with ([redacted: related project] §2.1: the answer must be the
        tail of the joint sequence).  Boundary handling per tokenizer family falls out of that
        rule instead of being hard-coded:

          BLIP2 (OPT BPE)  ' 1990' -> ['Ġ1990']: the space is inside the joint and the own suffix.
          LLaVA (LLaMA)    the space before a letter merges into the answer token ('▁K'),
                           but before a digit it stays a separate bare '▁' (id 29871) that the
                           stand-alone tokenisation also has -> kept, matching the joint form.
          Qwen             'assistant\n' + answer: no leading space anywhere.
        """
        joint = self._tok(text + sep + answer)
        own = self._tok(sep + answer)
        k = min(len(joint), len(own))
        while k > 0 and joint[-k:] != own[len(own) - k:]:
            k -= 1
        if k == 0:
            raise RuntimeError(f"answer tokenisation is not a suffix of the joint one: {answer!r}")
        return joint[-k:], joint

    def encode_answer(self, answer: str, sep: str = " ") -> list[int]:
        """Answer ids for a raw ``prompt + sep + answer`` string (used for tokeniser tests)."""
        return self.answer_and_joint("", answer, sep)[0]

    def encode_row(self, prompt: str, answer: str | None, fmt: str, n_image_tokens: int | None):
        """Token ids of one row: dict(ids, answer_start, answer_ids).

        When an answer is given the row is built from the *joint* tokenisation of
        ``prompt_text + sep + answer`` and the answer ids are its last ``len(enc)`` ids, so the
        model sees exactly the sequence a HF processor would build (tests/test_tokenization.py).
        """
        with_image = n_image_tokens is not None
        prefix, text, sep = self.text_parts(prompt, fmt, with_image)
        if answer is None:
            body, ans = self._tok(text), []
        else:
            ans, body = self.answer_and_joint(text, answer, sep)
            if not ans:
                raise ValueError(f"empty answer tokenisation for {answer!r}")
        ids = prefix + body
        if self.backbone == "qwen25vl" and with_image:
            k = ids.index(self.image_token_id)
            assert ids.count(self.image_token_id) == 1
            ids = ids[:k] + [self.image_token_id] * n_image_tokens + ids[k + 1:]
        return {"ids": ids, "answer_start": len(ids) - len(ans), "answer_ids": ans}

    # ---------------------------------------------------------------- images
    def process_images(self, images: Sequence) -> tuple[dict, list[int]]:
        """Vision inputs (on device, model dtype) and per-image number of LM image slots."""
        pil = [load_image(im) for im in images]
        if self.backbone == "qwen25vl":
            out = self.image_processor(images=pil, min_pixels=QWEN_MIN_PIXELS,
                                       max_pixels=QWEN_MAX_PIXELS, return_tensors="pt")
            thw = out["image_grid_thw"]
            n_tok = [int(t.prod()) // (self.merge ** 2) for t in thw]
            vis = {"pixel_values": out["pixel_values"], "image_grid_thw": thw}
        else:
            out = self.image_processor(images=pil, return_tensors="pt")
            vis = {"pixel_values": out["pixel_values"]}
            n = self.n_query if self.backbone == "blip2" else self.n_llava_img
            n_tok = [n] * len(pil)
        return vis, n_tok

    def estimate_image_tokens(self, img) -> int:
        """Number of LM image slots without decoding the image (header read only)."""
        if self.backbone == "blip2":
            return self.n_query
        if self.backbone == "llava":
            return self.n_llava_img
        if isinstance(img, Image.Image):
            w, h = img.size
        else:
            path = img if os.path.isabs(img) else os.path.join(VLKEB_IMG_DIR, img)
            with Image.open(path) as im:
                w, h = im.size
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
        rh, rw = smart_resize(h, w, factor=28, min_pixels=QWEN_MIN_PIXELS, max_pixels=QWEN_MAX_PIXELS)
        return (rh // 28) * (rw // 28)

    def row_length(self, image, prompt: str, answer: str | None, fmt: str = "vqa") -> int:
        """Token length of one row as :meth:`build_batch` will build it (image slots included)."""
        with_image = image is not None
        _, text, sep = self.text_parts(prompt, fmt, with_image)
        n = len(self._tok(text)) + (self.estimate_image_tokens(image) if with_image else 0)
        if answer is not None:
            n += len(self.answer_and_joint(text, answer, sep)[0])
        return n

    def plan_batches(self, rows: Sequence[tuple], max_seq_len: int, max_rows: int | None = None
                     ) -> tuple[list[list[int]], list[int]]:
        """Pack rows into batches: longest row first, padded length <= max_seq_len.

        ``rows[i] = (image_or_None, prompt, answer_or_None, fmt)`` (a 3-tuple is read as fmt
        "vqa"); every row of a batch must share the image/no-image status, which is guaranteed by
        construction here. Returns (batches, lengths).
        """
        lens = [self.row_length(*r) for r in rows]
        has = {r[0] is not None for r in rows}
        if len(has) > 1:
            raise ValueError("plan_batches needs a homogeneous image / text-only row list")
        order = sorted(range(len(rows)), key=lambda i: -lens[i])
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i in order:
            m = max(cur_max, lens[i])
            if cur and (m > max_seq_len or (max_rows is not None and len(cur) >= max_rows)):
                batches.append(sorted(cur))
                cur, cur_max = [i], lens[i]
            else:
                cur.append(i)
                cur_max = m
        if cur:
            batches.append(sorted(cur))
        return batches, lens

    # ----------------------------------------------------------------- batch
    def build_batch(self, images: Sequence | None, prompts: Sequence[str],
                    answers: Sequence[str] | None = None, fmt: str = "vqa",
                    max_len: int | None = None, vision: tuple[dict, list[int]] | None = None) -> dict:
        """Right-padded batch.

        images  : list of PIL images / paths, all None (text-only), or None
        prompts : raw questions (templated according to ``fmt``)
        answers : optional answers (teacher forcing)
        vision  : optional precomputed ``process_images(images)`` result
        Returns dict with ``inputs`` (model kwargs on device) and metadata tensors (on device):
          last_prompt_idx (B,), answer_start (B,), answer_len (B,), answer_ids (B,A) pad -100,
          pred_pos (B,A) (padding filled with last_prompt_idx), answer_mask (B,A) bool,
          labels (B,T) (-100 outside answers), seq_len (B,), n_image_tokens (B,), text_only.
        """
        B = len(prompts)
        if images is None:
            images = [None] * B
        if len(images) != B or (answers is not None and len(answers) != B):
            raise ValueError("images / prompts / answers length mismatch")
        has = [im is not None for im in images]
        if any(has) and not all(has):
            raise ValueError("mixed image and text-only rows in one batch; split them")
        with_image = B > 0 and all(has)
        if with_image:
            vis, n_tok = vision if vision is not None else self.process_images(images)
        else:
            vis, n_tok = {}, [None] * B
        rows = [self.encode_row(prompts[i], None if answers is None else answers[i], fmt, n_tok[i])
                for i in range(B)]
        if max_len is not None:
            # [not released: code that no released code calls; it served the related project (README, "What can be reproduced")]
            raise NotImplementedError("max_len: not part of the released loader")
        T = max(len(r["ids"]) for r in rows)
        A = max(1, max(len(r["answer_ids"]) for r in rows))
        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)
        ans_ids = torch.full((B, A), -100, dtype=torch.long)
        ans_mask = torch.zeros((B, A), dtype=torch.bool)
        pred_pos = torch.zeros((B, A), dtype=torch.long)
        a_start = torch.zeros(B, dtype=torch.long)
        a_len = torch.zeros(B, dtype=torch.long)
        seq_len = torch.zeros(B, dtype=torch.long)
        for i, r in enumerate(rows):
            n, s, a = len(r["ids"]), r["answer_start"], r["answer_ids"]
            input_ids[i, :n] = torch.tensor(r["ids"])
            attn[i, :n] = 1
            seq_len[i], a_start[i], a_len[i] = n, s, len(a)
            pred_pos[i, :] = s - 1
            if a:
                labels[i, s:s + len(a)] = torch.tensor(a)
                ans_ids[i, :len(a)] = torch.tensor(a)
                ans_mask[i, :len(a)] = True
                pred_pos[i, :len(a)] = torch.arange(s - 1, s - 1 + len(a))
        inputs = {"input_ids": input_ids, "attention_mask": attn}
        if with_image:
            inputs["pixel_values"] = vis["pixel_values"].to(self.dtype)
            if self.backbone == "qwen25vl":
                inputs["image_grid_thw"] = vis["image_grid_thw"]
                inputs["mm_token_type_ids"] = (input_ids == self.image_token_id).to(torch.int32)
        n_img = (input_ids == self.image_token_id).sum(1)
        if with_image:
            assert n_img.tolist() == list(n_tok), (n_img.tolist(), n_tok)
        else:
            assert int(n_img.sum()) == 0
        dev = self.device
        return {
            "inputs": {k: v.to(dev) for k, v in inputs.items()},
            "last_prompt_idx": (a_start - 1).to(dev),
            "answer_start": a_start.to(dev), "answer_len": a_len.to(dev),
            "answer_ids": ans_ids.to(dev), "answer_mask": ans_mask.to(dev),
            "pred_pos": pred_pos.to(dev), "labels": labels.to(dev),
            "seq_len": seq_len.to(dev), "n_image_tokens": n_img.to(dev),
            "text_only": not with_image, "fmt": fmt, "B": B, "T": T,
        }

    # --------------------------------------------------------------- forward
    def forward(self, batch: dict, logits_to_keep=None, stop_after_layer: int | None = None):
        """Run the model on ``batch['inputs']`` (no implicit no_grad; use_cache=False).

        logits_to_keep:
          None      -> logits for all positions (B, T, V)
          "answer"  -> per-row gather at batch['pred_pos'] -> (B, A, V)
          "last"    -> per-row gather at batch['last_prompt_idx'] -> (B, 1, V)
          LongTensor (B, K) -> per-row gather -> (B, K, V)
          int / 1-D LongTensor -> HF semantics (same positions for every row)
        stop_after_layer: run decoder blocks 0..l only and return None (use with capture()).
        """
        inputs = batch["inputs"]
        gather = None
        hf_keep = 0
        if isinstance(logits_to_keep, str):
            if logits_to_keep == "answer":
                gather = batch["pred_pos"]
            elif logits_to_keep == "last":
                gather = batch["last_prompt_idx"][:, None]
            else:
                raise ValueError(logits_to_keep)
        elif torch.is_tensor(logits_to_keep) and logits_to_keep.dim() == 2:
            gather = logits_to_keep.to(self.device)
        elif logits_to_keep is not None:
            hf_keep = logits_to_keep
        handles = []
        if gather is not None:
            def pre_hook(mod, args):
                h = args[0]
                idx = gather[:, :, None].expand(-1, -1, h.shape[-1])
                return (torch.gather(h, 1, idx),) + tuple(args[1:])
            handles.append(self.lm_head.register_forward_pre_hook(pre_hook))
        if stop_after_layer is not None:
            def stop_hook(mod, args, out):
                raise _StopForward
            handles.append(self.layer_module(stop_after_layer).register_forward_hook(stop_hook))
        try:
            if self.backbone == "blip2" and batch["text_only"]:
                out = self.m.language_model(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"],
                                            use_cache=False, logits_to_keep=hf_keep)
            else:
                out = self.m(**inputs, use_cache=False, logits_to_keep=hf_keep)
        except _StopForward:
            out = None
        finally:
            for h in handles:
                h.remove()
        return out

    # --------------------------------------------------------------- capture
    @staticmethod
    def _as_btd(x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        if x.dim() == 3:
            return x
        if x.dim() == 2 and x.shape[0] == B * T:   # OPT flattens the FFN to (B*T, d)
            return x.view(B, T, x.shape[-1])
        raise RuntimeError(f"unexpected activation shape {tuple(x.shape)} for B={B}, T={T}")

    def _selector(self, batch: dict, positions):
        B, T = batch["B"], batch["T"]
        ar = torch.arange(B, device=self.device)
        attn = batch["inputs"]["attention_mask"].bool()
        if positions is None or (isinstance(positions, str) and positions == "all"):
            return (lambda x: x), None
        if isinstance(positions, str):
            if positions == "last":
                idx = batch["last_prompt_idx"]
                return (lambda x: x[ar, idx]), None
            if positions == "answer":   # answer-predicting positions, flattened over valid ones
                mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
                pp, am = batch["pred_pos"], batch["answer_mask"]
                mask[ar[:, None].expand_as(pp)[am], pp[am]] = True
            # [not released: code that no released code calls; it served the related project (README, "What can be reproduced")]
            else:
                raise ValueError(positions)
            rows = ar[:, None].expand(B, T)[mask]
            return (lambda x: x[mask]), rows
        t = torch.as_tensor(positions, device=self.device)
        if t.dtype == torch.bool:
            rows = ar[:, None].expand(B, T)[t]
            return (lambda x: x[t]), rows
        if t.dim() == 1:
            return (lambda x: x[ar, t]), None
        if t.dim() == 2:
            return (lambda x: torch.gather(x, 1, t[:, :, None].expand(-1, -1, x.shape[-1]))), None
        raise ValueError(f"bad positions {positions!r}")

    @contextlib.contextmanager
    def capture(self, batch: dict, layers: Iterable[int], positions="last", module: str = "edit",
                io: Sequence[str] = ("in", "out"), dtype: torch.dtype | None = torch.float32,
                detach: bool = True):
        """Capture inputs/outputs of edit modules (``module='edit'``) or decoder blocks
        (``module='layer'``; output = residual stream) during the next forward(batch).

        positions: "last" (B,d) | "all"/None (B,T,d) | "answer"
                   (N,d), with ``cap.rows`` giving the row of each vector |
                   LongTensor (B,) -> (B,d) | LongTensor (B,K) -> (B,K,d) | BoolTensor (B,T) -> (N,d).
        """
        layers = list(layers)
        cap = Capture()
        sel, cap.rows = self._selector(batch, positions)
        B, T = batch["B"], batch["T"]
        get = self.edit_module if module == "edit" else self.layer_module

        def conv(x):
            x = sel(self._as_btd(x, B, T))
            if detach:
                x = x.detach()
            return x.to(dtype) if dtype is not None else x

        def make_hook(l):
            def hook(mod, args, output):
                if "in" in io:
                    cap.inp[l] = conv(args[0])
                if "out" in io:
                    cap.out[l] = conv(output[0] if isinstance(output, tuple) else output)
            return hook

        handles = [get(l).register_forward_hook(make_hook(l)) for l in layers]
        try:
            yield cap
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------ misc
    def info(self) -> dict:
        d = {"backbone": self.backbone, "dtype": str(self.dtype), "device": str(self.device),
             "n_layers": self.n_layers, "image_token_id": self.image_token_id,
             "edit_module": self.edit_module_name("{l}"), "layer_module": self.layer_module_name("{l}"),
             "qa_template": QA_TEMPLATE}
        if self.backbone == "llava":
            d["llava_image_sep"] = self.llava_image_sep
            d["n_image_tokens"] = self.n_llava_img
        if self.backbone == "blip2":
            d["n_image_tokens"] = self.n_query
        if self.backbone == "qwen25vl":
            d["min_pixels"], d["max_pixels"] = QWEN_MIN_PIXELS, QWEN_MAX_PIXELS
        if self.m is not None:
            d["d_in"], d["d_out"] = self.d_in, self.d_out
        return d

    def __repr__(self):
        return f"VLM({self.info()})"
