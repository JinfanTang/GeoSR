
# GeoGR²: Zero-Shot Geospatial Inference via Geostatistically-Guided Iterative Refinement with LLMs

[![EMNLP](https://img.shields.io/badge/EMNLP'26-Findings-orange)](https://arxiv.org/abs/xxxx.xxxxx) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/License-MIT-green)]()

This repository contains the official implementation of our EMNLP 2026 Findings paper: **GeoGR²: Zero-Shot Geospatial Inference via Geostatistically-Guided Iterative Refinement with LLMs**.

> **GeoGR²** (Geostatistically-Guided Iterative Refinement) is a zero-shot framework that leverages LLMs for geospatial inference through an iterative refinement pipeline: **GeoLLM** generates initial predictions, and **GeoGR²** refines them across multiple rounds using geostatistical guidance.



## Quickstart

> [!IMPORTANT]
> Before running the scripts, ensure you have a valid OpenAI API key. The default output directory for first-round predictions is `GeoLLM_results/`.

### 1. First-Round Prediction with GeoLLM

Generate initial spatial predictions using **GeoLLM**.

**Output folder:** `GeoLLM_results/` (default)

```bash
python GeoLLM.py \
  openai \
  sk-xxx \
  gpt-3.5-turbo-0125 \
  prompts/world_prompts.jsonl \
  "Infant Mortality Rate"
```

**Arguments:**

| Position | Description | Example |
|:---|:---|:---|
| `provider` | LLM provider | `openai` |
| `api_key` | Your API key | `sk-xxx` |
| `model` | Model name | `gpt-3.5-turbo-0125` |
| `prompt_file` | Path to prompt JSONL | `prompts/world_prompts.jsonl` |
| `task_name` | Name of the prediction task | `"Infant Mortality Rate"` |

---

### 2. Multi-Round Refinement with GeoGR²

Refine the initial predictions using **GeoGR²** with geostatistical guidance.

**Input files:**

| File | Purpose | Example Path |
|:---|:---|:---|
| `GeoLLM_result.csv` | First-round predictions from Step 1 | `GeoLLM_results/GeoLLM_result.csv` |
| `groundtruth.tif` | Ground-truth raster for the target variable | `data/povmap_global_subnational_infant_mortality_rates_v2_01.tif` |
| `anchoring.tif` | Raster for bias scoring (e.g., population density, economic indicators) | `data/ppp_2020_1km_Aggregated.tif` |

**Command:**

```bash
python GeoSR.py \
  GeoLLM_results/GeoLLM_result.csv \
  data/povmap_global_subnational_infant_mortality_rates_v2_01.tif \
  data/ppp_2020_1km_Aggregated.tif \
  "Infant Mortality Rate" \
  --api_key sk-xxx \
  --auxiliary_dir data \
  --output_dir ./your_output_dir \
  --model gpt-3.5-turbo-0125
```

**Optional arguments:**

| Flag | Description | Default |
|:---|:---|:---|
| `--api_key` | OpenAI API key | `None` |
| `--auxiliary_dir` | Directory containing auxiliary raster data | `data` |
| `--output_dir` | Directory to save refined results | `./your_output_dir` |
| `--model` | LLM model for refinement | `gpt-3.5-turbo-0125` |

Done! Refined results will be saved to `./your_output_dir/`.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex

```

---

## Contact

If you have any questions or suggestions, feel free to open an issue or contact the authors.
```
