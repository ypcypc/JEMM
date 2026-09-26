import json
import pathlib
import time

import numpy as np
import torch

from .contract import MAX_MM_TOKENS, MAX_SINGLE_TOKENS, encode_mm, encode_single, label_token_ids
from .systemone import answer, decode_images, single_requests

TEMPERATURE = 1.3480874159655591
THRESHOLD = 0.9872681877423998


def softmax(scores, temperature=1.0):
    values = np.asarray(scores, dtype=np.float64) / temperature
    return np.exp(values - np.logaddexp.reduce(values))


def _decision_config(adapter):
    path = pathlib.Path(adapter) / "decision_config.json"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        try:
            path = pathlib.Path(hf_hub_download(adapter, "decision_config.json"))
        except (OSError, ValueError):
            return {}
    config = json.loads(path.read_text(encoding="utf-8"))
    return {k: config[k] for k in ("temperature", "threshold", "mm_temperature") if k in config}


class DecisionModel:
    def __init__(self, model, tokenizer, name="jemm", temperature=TEMPERATURE, threshold=THRESHOLD,
                 max_tokens=MAX_SINGLE_TOKENS, processor=None, mm_temperature=None, max_mm_tokens=MAX_MM_TOKENS):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.labels = label_token_ids(tokenizer)
        self.name = name
        self.temperature = temperature
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.processor = processor
        self.mm_temperature = temperature if mm_temperature is None else mm_temperature
        self.max_mm_tokens = max_mm_tokens
        self.device = next(model.parameters()).device

    @classmethod
    def from_pretrained(cls, adapter, base=None, device="cuda:0", **kwargs):
        from peft import PeftConfig, PeftModel
        from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
        base = base or PeftConfig.from_pretrained(adapter).base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base)
        try:
            processor = AutoProcessor.from_pretrained(base)
        except OSError:
            processor = None
        model = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map={"": device},
                                                                attn_implementation="sdpa")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        options = {"name": pathlib.PurePosixPath(str(adapter).replace("\\", "/")).name, "processor": processor,
                   **_decision_config(adapter), **kwargs}
        return cls(model, tokenizer, **options)

    def _inputs(self, request, images):
        if not images:
            return {"input_ids": torch.tensor([encode_single(self.tokenizer, request, self.max_tokens)], dtype=torch.long)}
        if self.processor is None:
            raise ValueError("images need the base model's image processor (preprocessor_config.json)")
        return encode_mm(self.processor, request, images, self.max_mm_tokens)

    def scores(self, request, images=None):
        """Raw label logits of the candidates, and the prompt length."""
        dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(self.device, dtype=dtype) if k == "pixel_values" else v.to(self.device)
                  for k, v in self._inputs(request, images).items()}
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
        return logits[self.labels[:len(request["candidates"])]].float().cpu().tolist(), inputs["input_ids"].shape[1]

    def probabilities(self, request, images=None):
        scores, _ = self.scores(request, images)
        temperature = self.mm_temperature if images else self.temperature
        return {c["id"]: float(p) for c, p in zip(request["candidates"], softmax(scores, temperature))}

    def _decide(self, state, questions, images=None):
        answers, tokens = {}, 0
        temperature = self.mm_temperature if images else self.temperature
        for qid, kind, request in single_requests(state, questions):
            scores, n = self.scores(request, images)
            probs = {c["id"]: float(p) for c, p in zip(request["candidates"], softmax(scores, temperature))}
            answers[qid] = answer(kind, probs, max(probs, key=probs.get))
            tokens += n
        return answers, tokens

    def decide(self, state, questions, images=None):
        """images: optional PIL images shown to every question."""
        return self._decide(state, questions, images)[0]

    def respond(self, body):
        """A full POST /v1/systemone response for a request body."""
        started = time.perf_counter()
        images = decode_images(body.get("images"))
        answers, tokens = self._decide(body.get("state", ""), body.get("questions"), images)
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0, "latency_ms": (time.perf_counter() - started) * 1000}}
