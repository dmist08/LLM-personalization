---
description: Re-run full evaluation and regenerate results table
---
Run the evaluation pipeline:
// turbo-all
1. Run: `python -m src.pipeline.evaluate --compute-report`
2. Show me the full results table from `outputs/evaluation/result_table.txt`
3. Print the key pass/fail check: does cold-start beat baseline on sparse authors?
4. Compare to previous run if `outputs/evaluation/result_table.json` already exists — show what changed
5. Show the computational cost report from `outputs/evaluation/compute_cost_report.json`
