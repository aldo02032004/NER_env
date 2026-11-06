# Indonesian Political-Economic NER

Named-entity recognition for Indonesian social-media posts about politics and the economy. GLiNER finds the entities, Gemini checks the uncertain ones, and every checked document becomes training data for a smaller IndoBERT model, so LLM calls go down over time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aldo02032004/NER_env/blob/main/notebooks/01_label_entities.ipynb)

**Status:** stage 2 (labelling with GLiNER + Gemini) is done and tested. Stage 3 (training and evaluating IndoBERT on the collected data) is next.

## Entities

`tokoh` (person) · `instansi_pemerintah` (government agency) · `partai` (political party) · `perusahaan` (company) · `kebijakan` (policy or programme) · `indikator_ekonomi` (economic indicator) · `nilai_uang` (money amount) · `lokasi` (location)

## How it works

```
spreadsheet → clean text → GLiNER → routing → Gemini check → master JSONL (training data)
                                                           → output/<project>_entities.csv
```

1. **Clean**: build one text per post, flag media accounts, retweets, very short posts and duplicates. No rows are dropped; each row gets a `siap_ner` (ready for NER) flag.
2. **Predict**: GLiNER runs in batches. Long texts are split into overlapping chunks, and predictions are cached per document.
3. **Route**: only documents worth checking go to Gemini:

   | route_reason | Sent when |
   |---|---|
   | `low_score` | an entity scores below the review threshold |
   | `no_entity_capital` | no entity, but a capitalised word appears mid-sentence |
   | `no_entity_sample` | no entity, picked in a small random sample |
   | `random_sample` | high-scoring document, picked in a random sample (the honest accuracy check) |

4. **Validate**: Gemini corrects the entities in batches. A document enters the training data only if every entity is found exactly in the text and no name has two labels. Runs resume after quota limits, and documents already finished are never re-sent.
5. **Report and export**: precision and recall of GLiNER against Gemini per route, and a per-project table with the entities in one column per label.

## Quick start (Google Colab)

1. Create a Gemini API key at <https://aistudio.google.com/apikey> and save it in Colab **Secrets** as `GOOGLE_API_KEY` (enable *Notebook access*).
2. Open the notebook with the badge above and select a **T4 GPU** runtime.
3. Run cell ①, fill in the form in cell ② (project name and spreadsheet links), then **Runtime → Run cell and below**.

Files are written to Google Drive (locations set in `paths` in the config):

| File | Contents |
|---|---|
| `master_ner.jsonl` | training data shared by all projects (format in [PROJECT_NOTES.md](PROJECT_NOTES.md)) |
| `checkpoints/preds_<project>_<model>_t<threshold>.jsonl` | cached GLiNER predictions |
| `checkpoints/llm_log_<project>.jsonl` | Gemini log per document, used to resume |
| `output/<project>_entities.pkl` / `.csv` | original columns + `doc_id`, `text`, `entities_<label>`, `ner_source` |

## Tests

Requires Python ≥ 3.11.

```bash
pip install -r requirements.txt
python -m pytest -q
```

The 90 tests cover every pure function (cleaning, formats, schema, routing, config, LLM parsing) without a GPU or API key.

## Project structure

```
NER_env/
├── ner/                         # all logic, importable and tested
├── notebooks/01_label_entities.ipynb
├── tests/
├── PROJECT_NOTES.md             # labels, master data format, rules
└── requirements.txt
```

## Module reference

### `ner.config`: config JSON on Drive

```python
from ner.config import load_config, save_config
cfg = load_config("/content/drive/MyDrive/ner/config.json")   # missing -> defaults are written first
cfg["batch_size"] = 20
save_config(cfg, "/content/drive/MyDrive/ner/config.json")   # validated before writing
```

An invalid config raises one `ValueError` that lists **every** problem, including misspelt keys. Keys added in newer versions are filled with defaults while your edited values are kept. Old default GLiNER prompts that were never edited are upgraded to English automatically.

| Key | Meaning |
|---|---|
| `labels` | label list (draft) |
| `label_descriptions` | per label: `deskripsi`, `contoh_positif`, `contoh_negatif`, `aturan`, all sent to Gemini |
| `gliner_label_prompts` | per label: the text sent to GLiNER. **Use English** ("person", "political party", …): GLiNER v2.1 does not understand Indonesian labels. Changing a prompt re-runs predictions (separate cache). |
| `predict_threshold` | minimum GLiNER score kept |
| `ner_review_threshold` | per model: documents with an entity below this score go to Gemini (`low_score`) |
| `review_sample_rate` | share of high-scoring documents still checked (`random_sample`) |
| `no_entity_sample_rate` | share of entity-free documents without mid-sentence capitals that are checked (`no_entity_sample`) |
| `llm_provider` | `gemini_api` (key in Colab Secrets) or `colab` (`google.colab.ai`, very small quota) |
| `llm_model` | Gemini model name, default `gemini-2.5-flash` |
| `llm_budget_per_run` | maximum documents sent per run (`null` = until the quota runs out) |
| `aliases` | `{project: {variant: canonical name}}`, used only in the export |

### `ner.cleaning`: raw text → master text

```python
from ner.cleaning import build_raw_text, text_for_ner
raw = build_raw_text(row["Headline"], row["Mentions"])
text = text_for_ner(raw)   # the only path to the master `text`
```

### `ner.schema`: master records and JSONL

```python
from ner.schema import make_record, append_jsonl
rec = make_record(text, spans, labels=cfg["labels"], project="my_project",
                  label_source="gemini", route_reason="low_score",
                  model_version=cfg["active_model"], schema_version=cfg["schema_version"])
append_jsonl(cfg["paths"]["master_data"], [rec], allowed_labels=cfg["labels"])
```

- `doc_id` is a hash of the normalised text, so texts that differ only in punctuation or case count as one document.
- `append_jsonl` is safe to repeat: existing `doc_id`s are skipped, invalid records raise `ValueError`, and a last line cut off by a crashed runtime is dropped with a warning.

### `ner.formats`: entity conversions

```python
from ner.formats import entities_dict_to_spans, spans_to_bio, spans_to_gliner
spans, missing = entities_dict_to_spans(text, {"tokoh": ["Prabowo"], "lokasi": ["Rusia"]})
tokens, tags, n_misaligned = spans_to_bio(text, spans)   # BIO tags with the GLiNER tokenizer
item, n_misaligned = spans_to_gliner(text, spans)
```

Matching is case-insensitive and whole-word, and the longer name wins on overlap. `n_misaligned > 0` means a span did not line up with token boundaries and was widened. For evaluation, convert gold and predicted spans with `spans_to_bio` on the same text and score with seqeval.

### `ner.io`, `ner.predict`, `ner.routing`, `ner.llm`

- `ner.io`: `load_sheet` merges several Sheets/Drive links, `prepare_frame` adds the flags above, and `build_export` builds the output table. Duplicates inherit the entities of the original post, and unchecked GLiNER entities are kept only above the review threshold.
- `ner.predict`: `predict_all` runs GLiNER with chunking and a per-document cache.
- `ner.routing`: `route_all` assigns each document a `route_reason`. Samples are chosen from a hash of `doc_id`, so the choice is the same on every run.
- `ner.llm`: `review_all` sends batches to Gemini. A batch that fails to parse twice is retried one document at a time, and a document that still fails is marked `failed`. The queue alternates between route reasons, and `low_score` starts with the lowest scores. `summarize_log` reports, per route, how many model entities Gemini confirmed, removed or added, together with precision and recall.

## License

MIT, see [LICENSE](LICENSE).
