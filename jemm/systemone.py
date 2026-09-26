import base64
import binascii
import io
import json

from PIL import Image

from .contract import MAX_MM_TOKENS

MAX_IMAGES = 4
# One image token covers 32x32 pixels, so a larger image cannot fit the prompt.
MAX_IMAGE_PIXELS = MAX_MM_TOKENS * 32 * 32
IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}


def _description(spec):
    if isinstance(spec, dict):
        return str(spec.get("description", ""))
    return "" if spec is None else str(spec)


def _action(spec):
    if isinstance(spec, dict) and isinstance(spec.get("action"), dict):
        return spec["action"]
    return {}


def candidates_for(question):
    kind = question.get("type", "choice")
    criteria = question.get("criteria")
    if kind == "noul":
        crit = criteria if isinstance(criteria, dict) else {}
        return [{"id": "yes", "description": _description(crit.get("true")) or "yes", "action": {}},
                {"id": "no", "description": _description(crit.get("false")) or "no", "action": {}}]
    if kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score question needs a list of at least two levels")
        return [{"id": str(i), "description": _description(level) or f"level {i}", "action": {}}
                for i, level in enumerate(criteria)]
    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice question needs a criteria map with at least two options")
        return [{"id": str(cid), "description": _description(c), "action": _action(c)} for cid, c in criteria.items()]
    raise ValueError(f"unsupported question type: {kind!r}")


def single_requests(state, questions):
    """Yields (question_id, type, request) with one request per question."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("no questions")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    for qid, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"question {qid!r} is not an object")
        request = {"state": state, "question": str(question.get("instructions", "")), "candidates": candidates_for(question)}
        yield qid, question.get("type", "choice"), request


def _decode_image(index, item):
    if not isinstance(item, str):
        raise ValueError(f"image {index} is not a base64 string")
    if item.startswith("data:"):
        head, _, item = item.partition(",")
        if not head.endswith(";base64"):
            raise ValueError(f"image {index} is not a base64 data URL")
    try:
        with Image.open(io.BytesIO(base64.b64decode("".join(item.split()), validate=True))) as image:
            if image.format not in IMAGE_FORMATS:
                raise ValueError(f"image {index} is not PNG, JPEG or WebP")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"image {index} exceeds {MAX_IMAGE_PIXELS} pixels")
            return image.convert("RGB")
    except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"image {index} could not be decoded") from exc


def decode_images(items):
    """RGB images from a list of base64 strings or data URLs."""
    if items is None:
        return []
    if not isinstance(items, list) or len(items) > MAX_IMAGES:
        raise ValueError(f"images must be a list of at most {MAX_IMAGES} base64 strings")
    return [_decode_image(i, item) for i, item in enumerate(items)]


def answer(kind, probabilities, choice):
    confidence = max(probabilities.values())
    if kind == "noul":
        return {"type": "noul", "noul": probabilities["yes"], "probabilities": probabilities, "confidence": confidence}
    if kind == "score":
        return {"type": "score", "probabilities": probabilities, "confidence": confidence,
                "expected_value": sum(float(k) * p for k, p in probabilities.items())}
    return {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": confidence}
