# JEMM

**Like [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), but multimodal and open-weight.**

JEMM picks one candidate per question and returns a probability for every candidate, from text or text plus a screenshot, on your own GPU.

JEMM stands for Judgment Engine for MultiModal decisions. This repository serves [JEMM](https://huggingface.co/MaestroYan/JEMM) over HTTP with a Jev-style `POST /v1/systemone` endpoint; the model card shows usage without this server. Needs one CUDA GPU with at least 64 GB. Not affiliated with TypeSafe AI.

![JEMM vs open Jev-like models](assets/landscape.png)

![Accuracy of JEMM vs Jev 1.13](assets/accuracy.png)

![Latency of JEMM vs Jev 1.13](assets/latency.png)

## Usage

```bash
pip install git+https://github.com/ypcypc/JEMM
python -m jemm.serve --adapter MaestroYan/JEMM --port 8790
```

```bash
curl -s http://127.0.0.1:8790/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "User: Will it rain in Paris tomorrow?",
  "questions": {"tool": {"type": "choice", "instructions": "Which tool should handle the request?",
    "criteria": {"get_weather": "Weather forecast for a city", "search_flights": "Flight search", "none": "No tool applies"}}}
}'
```

- `type`: `choice` (2 to 32 options), `noul` (`{"true": ..., "false": ...}`) or `score` (a list of levels).
- `images` (optional): up to 4 base64 PNG, JPEG or WebP screenshots, seen by every question.
- Each answer has `probabilities` and `confidence`. Treat `confidence` below `threshold` in `decision_config.json` as undecided.

Apache-2.0.
