# Does the agent know when its order is wrong?

Code, cached model responses and results for

> H. Lu. *Does the agent know when its order is wrong? Verbalised and trace-based confidence of large language model agents in inventory decisions.* Submitted to Decision Support Systems, 2026.

Six language models (Qwen2.5-7B, Llama-3.1-8B, Gemma-4-E4B, Phi-4-mini, GPT-6 Luna, Claude Haiku 5.5) act as newsvendor agents on three public demand panels (NHS England prescribing, Australian PBS, FreshRetailNet-50K). Each model is asked for the same decisions in four ways (basic, informed, agentic, quantile), and the paper asks whether the agent's stated confidence, or its behaviour, reveals when its order is wrong.

## What is here

| Path | Content |
|---|---|
| `UTD24_LLM_inventory_calibration_mac.ipynb` | The full pipeline as a notebook: data, prompts, model calls, experiments E1-E5, tables and figures. |
| `run_mac.py` | The same pipeline as a plain script (generated from the notebook by `nb2py.py`). |
| `utd24_runs/cache/*.jsonl` | Every model response used in the paper, keyed by a hash of (model, prompt, temperature, seed). With the cache in place the pipeline re-runs **without Ollama or any API key**: every prompt the paper issued is answered from the cache, so the paper's numbers are reproduced as long as the decision sets are unchanged (they are also provided below). |
| `utd24_runs/data/decisions_*_full.pkl` | The exact decision sets (series, split, 24-period history, realised demand, SAA quantile) for the three datasets. |
| `utd24_runs/data/freshretail_panel.csv`, `practice_groups_imd.csv` | Derived FreshRetail panel and the NHS practice-to-IMD-quintile mapping. |
| `utd24_runs/results/` | Per-decision outputs for every model and condition (`*_decisions_{basic,informed,agentic,quantile}_*.csv`), experiment tables (`*_E1_table.csv` ... `*_E5_quantile.csv`), and the derived analyses reported in the paper (`analysis2_confidence_collapse.csv` ... `analysis9_pipeline.csv`). |
| `paper/` | Figure PDFs and `make_fig_ladder.py` (Figure 3). |

Raw NHS and PBS files are not included (about 250 MB); the code downloads them from the NHSBSA open-data API and pbs.gov.au on first run. FreshRetailNet-50K is loaded from the Hugging Face hub.

## Reproduce

```bash
pip install -r requirements.txt
python3 run_mac.py nhs all          # also: pbs | freshretail
python3 run_mac.py pbs all          # PBS in thousands (paper default); PBS_UNITS=raw for raw units
python3 run_mac.py freshretail all
```

Each run writes `utd24_runs/results/{dataset}_full_*.csv` and the figures. Every prompt is served from `utd24_runs/cache/`, so no model is called unless a prompt is new (for example after a change to the data or the prompt text). To run new models or prompts, install [Ollama](https://ollama.com) and pull `qwen2.5:7b-instruct llama3.1:8b gemma4 phi4-mini`, and put `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in a `.env` file next to the script. API spend is capped by `LLM_BUDGET_USD` in the configuration block; the paper's total was USD 5.44.

Setting: newsvendor with unit costs b = 9, h = 1 (target service level tau = 0.9), 24-period history, 6 calibration and 6 test periods per series. Open-weight models run 4-bit through Ollama at temperature 0; the two API models at temperature 0 with reasoning disabled for GPT-6 Luna.

## Data sources

- NHS Business Services Authority, English Prescribing Dataset (Open Government Licence v3.0).
- Services Australia / Department of Health, PBS date-of-supply statistics (CC BY 4.0).
- FreshRetailNet-50K, Dingdong Inc., Hugging Face (`Dingdong-Inc/FreshRetailNet-50K`).
- Index of Multiple Deprivation 2019 (MHCLG) via postcodes.io for practice groups.

## Licence

MIT for the code. Data remain under their original licences.
