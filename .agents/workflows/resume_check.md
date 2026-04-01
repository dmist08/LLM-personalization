---
description: Check which long-running jobs are complete and what needs to resume
---
Check these files and report status (exists / missing / partial):
// turbo-all
1. `data/interim/indian_agnostic_headlines.csv` — expected ~6,500 rows
2. `data/interim/lamp4_agnostic_headlines.csv` — expected ~12,500 rows
3. `author_vectors/indian/layer_21/` — expected ~42 .npy files
4. `author_vectors/lamp4/layer_21/` — expected 200+ .npy files
5. `checkpoints/qlora/final/` — QLoRA training complete?
6. `checkpoints/qlora/merged/` — Merged model available?
7. `outputs/baselines/rag_and_base_outputs.jsonl` — RAG baselines done?
8. `outputs/stylevector_outputs.jsonl` — StyleVector inference done?
9. `outputs/cold_start_outputs.jsonl` — Cold-start inference done?

For each partial file: show current row/file count vs. expected.
