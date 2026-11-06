# Project notes

## Goal

Extract entities from Indonesian social-media posts about politics and the economy.
GLiNER predicts entities, a local LLM (Gemma 2 9B SahabatAI, GGUF via llama-cpp) checks the uncertain ones, and the checked results become training data for a smaller IndoBERT model. As that model improves, fewer LLM calls are needed, which makes the pipeline faster and cheaper.

## Labels (schema_version 1, draft)

`tokoh`, `instansi_pemerintah`, `partai`, `perusahaan`, `kebijakan`, `indikator_ekonomi`, `nilai_uang`, `lokasi`

Descriptions, positive and negative examples, and rules for each label live in `ner/config.py` (`LABEL_DESCRIPTIONS`) and are sent to the LLM.

## Master data format (JSONL, one document per line)

```json
{"doc_id": "hash of the text", "project": "str", "text": "output of clean_for_ner",
 "entities": [[start_char, end_char, "label"], ...],
 "labels": ["labels asked during annotation"], "schema_version": 1,
 "label_source": "llm | manual | gemini (older data)",
 "route_reason": "low_score | random_sample | no_entity_capital | no_entity_sample",
 "model_version": "str", "run_date": "YYYY-MM-DD"}
```

Spans are character offsets into `text`. Other formats (BIO, GLiNER) are always derived from this one and never stored as the source of truth.

## Rules

- Cleaning for NER keeps capital letters and punctuation.
- A document whose LLM entities cannot be found exactly in the text does **not** enter the training data.
- Gold data is never used for training.
- Train/validation splits are made per project or per time period, never randomly per row.
- Main metric: entity-level F1 (seqeval), reported per label.
- LLM validation resumes from a per-document log after a disconnect or when `llm_budget_per_run` is reached.
- Text cleaning (`clean_for_ner`, `clean_for_bert`, `is_media_account`) comes from the [GREAT-Tools](https://github.com/azmkto/GREAT-Tools) package, pinned to a commit, and is not copied into this repo.
