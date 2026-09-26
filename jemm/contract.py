import json
import re
import string

LABELS = list(string.ascii_uppercase) + list("012345")
SYSTEM = ("Choose the best available candidate for the question using only the supplied state. "
          "Return exactly one candidate label.")
MAX_SINGLE_TOKENS = 8192
MAX_MM_TOKENS = 3072
MAX_BUNDLE_TOKENS = 4096
MAX_BUNDLE_QUESTIONS = 8

_WS = re.compile(r"\s+")
_STANDARD_QUESTION = re.compile(r"^For query q\d+, Select the single appropriate tool, or NO_TOOL if none is suitable\.$")
_BUNDLE_INSTRUCTION = "For each query choose the single appropriate tool, or NO_TOOL if none is suitable."


def _flat(text):
    return _WS.sub(" ", text or "").strip()


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _param_names(params):
    if not isinstance(params, dict):
        return []
    props = params.get("properties", params) if isinstance(params.get("properties", None), dict) else params
    required = set(params.get("required", [])) if isinstance(params.get("required", []), list) else set()
    if not isinstance(props, dict):
        return []
    names = []
    for key, spec in props.items():
        if key in {"properties", "required", "type"} and not isinstance(spec, dict):
            continue
        piece = key
        if isinstance(spec, dict):
            if key in required:
                piece += "*"
            default = spec.get("default", None)
            if default not in (None, ""):
                piece += "=" + _flat(str(default))[:24]
        names.append(piece)
    return names


def _render_tool(description):
    try:
        tool = json.loads(description)
    except (json.JSONDecodeError, TypeError):
        return _flat(description)
    if not isinstance(tool, dict) or "name" not in tool:
        return _flat(description)
    text = _flat(str(tool.get("name", "")))
    summary = _flat(str(tool.get("description", "")))
    if summary:
        text += " \u2014 " + summary
    names = _param_names(tool.get("parameters"))
    return text + (" | params: " + ", ".join(names) if names else "")


def render_candidate(candidate):
    if "tool_name" in candidate.get("action", {}):
        return _render_tool(candidate["description"])
    return _flat(candidate["description"])


def _check_candidates(candidates):
    if not 2 <= len(candidates) <= len(LABELS):
        raise ValueError("UNSUPPORTED: candidate count")
    ids = [c["id"] for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate IDs")


def single_messages(request):
    candidates = request["candidates"]
    _check_candidates(candidates)
    state = request["state"] if isinstance(request["state"], str) else _dump(request["state"])
    lines = [f"{LABELS[i]}) {render_candidate(c)}" for i, c in enumerate(candidates)]
    user = ("State:\n" + state.strip() + "\n\nQuestion: " + _flat(request["question"]) +
            "\n\nCandidates:\n" + "\n".join(lines) + "\n\nAnswer with exactly one candidate label.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _bundle_state(state):
    parsed = state
    if isinstance(state, str):
        try:
            parsed = json.loads(state)
        except (json.JSONDecodeError, TypeError):
            return state.strip()
    if isinstance(parsed, dict) and isinstance(parsed.get("queries"), dict) and parsed["queries"]:
        lines = []
        for qid, messages in parsed["queries"].items():
            if isinstance(messages, list):
                text = " ".join(_flat(str(m.get("content", ""))) if isinstance(m, dict) else _flat(str(m)) for m in messages)
            else:
                text = _flat(str(messages))
            lines.append(f"{qid}: {text}")
        return "\n".join(lines)
    return parsed.strip() if isinstance(parsed, str) else _dump(parsed)


def _bundle_candidate(candidate):
    if candidate.get("action", {}).get("operation") == "NO_TOOL":
        return "NO_TOOL"
    return render_candidate(candidate)


def _marker(question_id):
    return f"Answer {question_id}:"


def _question_block(question):
    _check_candidates(question["candidates"])
    qid = question["question_id"]
    text = _flat(question["question"])
    head = f"{qid}:" if _STANDARD_QUESTION.match(text) else f"{qid}: {text}"
    lines = [f"{LABELS[i]}) {_bundle_candidate(c)}" for i, c in enumerate(question["candidates"])]
    return head + "\n" + "\n".join(lines) + "\n" + _marker(qid)


def bundle_text(request):
    questions = request["questions"]
    if not 2 <= len(questions) <= MAX_BUNDLE_QUESTIONS:
        raise ValueError("UNSUPPORTED: bundle size")
    blocks = [_question_block(q) for q in questions]
    text = ("Queries:\n" + _bundle_state(request["state"]) + "\n\n" + _BUNDLE_INSTRUCTION + "\n\n"
            + "\n\n".join(blocks) + "\n\nEach answer is exactly one candidate label.")
    return text, [_marker(q["question_id"]) for q in questions]


def _chat_tokens(tokenizer, messages, generation_prompt=True):
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_prompt,
                                           enable_thinking=False, preserve_thinking=False, return_dict=False)
    if not isinstance(tokens, list) or any(not isinstance(t, int) for t in tokens):
        raise TypeError("Tokenizer must return a flat integer token sequence")
    return tokens


def label_token_ids(tokenizer):
    return [tokenizer.encode(x, add_special_tokens=False)[0] for x in LABELS]


def encode_single(tokenizer, request, max_tokens=MAX_SINGLE_TOKENS):
    tokens = _chat_tokens(tokenizer, single_messages(request))
    if len(tokens) > max_tokens:
        raise ValueError("UNSUPPORTED: input exceeds token budget")
    return tokens


def mm_messages(request, n_images):
    system, user = single_messages(request)
    content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": user["content"]}]
    return [system, {"role": "user", "content": content}]


def encode_mm(processor, request, images, max_tokens=MAX_MM_TOKENS):
    """Model inputs for a request with the screenshots placed before the text."""
    if not images:
        raise ValueError("no images")
    text = processor.apply_chat_template(mm_messages(request, len(images)), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False, preserve_thinking=False)
    enc = processor(text=[text], images=list(images), return_tensors="pt")
    if enc["input_ids"].shape[1] > max_tokens:
        raise ValueError("UNSUPPORTED: input exceeds token budget")
    return {k: enc[k] for k in ("input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids")}


def encode_bundle(tokenizer, request, max_tokens=MAX_BUNDLE_TOKENS):
    """Returns the prompt token ids and, per question, the position whose next-token logits hold its answer."""
    text, markers = bundle_text(request)
    full = _chat_tokens(tokenizer, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
    if len(full) > max_tokens:
        raise ValueError("UNSUPPORTED: input exceeds token budget")
    header = len(_chat_tokens(tokenizer, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ""}], False))
    user = tokenizer.encode(text, add_special_tokens=False)
    start = next((i for i in range(max(0, header - 8), min(len(full), header + 8) + 1) if full[i:i + len(user)] == user), None)
    if start is None:
        raise ValueError("Bundle user text is not token-stable inside the chat template")
    positions, cursor = [], 0
    for marker in markers:
        cursor = text.index(marker, cursor) + len(marker)
        prefix = tokenizer.encode(text[:cursor], add_special_tokens=False)
        if user[:len(prefix)] != prefix:
            raise ValueError("Marker boundary is not token-stable")
        positions.append(start + len(prefix) - 1)
    return full, positions
