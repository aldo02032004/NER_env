# Indonesian Political-Economic NER

Finds entities in Indonesian social-media posts about politics and the economy. GLiNER tags the posts, a local LLM (Gemma 2 9B SahabatAI, GGUF) checks the uncertain ones, and every checked post becomes training data for a smaller IndoBERT model, so fewer LLM calls are needed over time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aldo02032004/NER_env/blob/main/notebooks/01_label_entities.ipynb)

**Status:** stage 2 (labelling with GLiNER + local LLM) works and is tested. Stage 3 (training IndoBERT) is next.

## Entities

`tokoh` (person) · `instansi_pemerintah` (government agency) · `partai` (political party) · `perusahaan` (company) · `kebijakan` (policy or programme) · `indikator_ekonomi` (economic indicator) · `nilai_uang` (money amount) · `lokasi` (location)

## How it works

```
spreadsheet → clean → GLiNER → routing → LLM check → master JSONL (training data)
                                                    → output/<project>_entities.csv
```

1. **Clean**: one text per post. Media accounts, retweets, very short posts and duplicates are flagged, not dropped.
2. **Predict**: GLiNER runs in batches, long texts are split into chunks, and results are cached.
3. **Route**: only posts worth checking go to the LLM:

   | route_reason | Sent when |
   |---|---|
   | `low_score` | an entity scores below the review threshold |
   | `no_entity_capital` | no entity, but a capitalised word appears mid-sentence |
   | `no_entity_sample` | no entity, picked in a small random sample |
   | `random_sample` | high score, picked in a random sample (the honest accuracy check) |

4. **Validate**: the LLM corrects the entities, and its answer is forced into a JSON schema. A post enters the training data only if every entity is found exactly in the text and no name has two labels.
5. **Export**: GLiNER precision/recall against the LLM per route, plus one table per project.

Every step can resume: after a disconnect, run the notebook again and finished work is skipped.

## Quick start (Google Colab)

1. Open the notebook with the badge above and pick a **T4 GPU** runtime. No API key is needed.
2. Run cell 1, fill in the form in cell 2, then **Runtime → Run cell and below**.

Output on Google Drive (`paths` in the config):

| File | Contents |
|---|---|
| `master_ner.jsonl` | training data for all projects (format in [PROJECT_NOTES.md](PROJECT_NOTES.md)) |
| `checkpoints/preds_<project>_<model>_t<threshold>_p<prompt hash>.jsonl` | cached GLiNER predictions |
| `checkpoints/llm_log_<project>.jsonl` | LLM log per post, used to resume |
| `output/<project>_entities.pkl` / `.csv` | original columns + `doc_id`, `text`, `entities_<label>`, `ner_source` |

## Config

`config.json` on Drive is created with defaults on the first run. It is checked on every load, and all problems are reported at once. Keys added later are filled in automatically.

| Key | Meaning |
|---|---|
| `labels`, `label_descriptions` | labels and their description, examples and rules (sent to the LLM) |
| `gliner_label_prompts` | text sent to GLiNER per label, **in English** ("person", "political party", …) |
| `predict_threshold` | minimum GLiNER score kept |
| `ner_review_threshold` | per model: below this score a post goes to the LLM |
| `review_sample_rate`, `no_entity_sample_rate` | share of posts checked at random |
| `batch_size` | posts per LLM call (default 10, must fit in `llm_n_ctx`) |
| `llm_model_repo`, `llm_model_file`, `llm_n_ctx` | GGUF model on Hugging Face and its context length |
| `llm_budget_per_run` | max posts sent to the LLM per run (`null` = all) |
| `aliases` | `{project: {variant: canonical name}}`, export only |

## Code

| Module | Does |
|---|---|
| `ner.config` | load, validate and save the config |
| `ner.cleaning` | raw post → master `text` (cleaners come from [GREAT-Tools](https://github.com/azmkto/GREAT-Tools)) |
| `ner.io` | load spreadsheets, add flags, build the export table |
| `ner.predict` | GLiNER with chunking and cache |
| `ner.routing` | pick which posts go to the LLM (same choice on every run) |
| `ner.llm` | load the GGUF model, send batches, write training data, summarise |
| `ner.schema` | master record format and safe JSONL read/write |
| `ner.formats` | convert spans to BIO and GLiNER format and back |

Tests need Python ≥ 3.11 and no GPU:

```bash
pip install -r requirements.txt
python -m pytest -q
```

## License

MIT, see [LICENSE](LICENSE).
