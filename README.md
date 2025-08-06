# GeoSR Quick Start

Two short steps to run **GeoLLM + GeoSR** for multi-round spatial prediction.

---

## 1️⃣ First-Round Prediction with GeoLLM

- **Output folder:** `GeoLLM_results` (default)  
- **Command:**

```bash
python GeoLLM.py \
  openai \
  sk-xxx \
  gpt-3.5-turbo-0125 \
  prompts/world_prompts.jsonl \
  "Infaint Mortality Rate"
```

  ## 2️⃣ Multi-Round Refinement with GeoSR
  | File                | Purpose                                                     | Example                                                           |
| ------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| `GeoLLM_result.csv` | First-round predictions from step 1                         | `GeoLLM_results/GeoLLM_result.csv`                                |
| `groundtruth.tif`   | Ground-truth raster for the task                            | `data/povmap_global_subnational_infant_mortality_rates_v2_01.tif` |
| `anchoring.tif`     | Raster for bias scoring (population density, economy, etc.) | `data/ppp_2020_1km_Aggregated.tif`                                |

Command:
```bash
python GeoSR.py \
  GeoLLM_results/GeoLLM_result.csv \
  data/povmap_global_subnational_infant_mortality_rates_v2_01.tif \
  data/ppp_2020_1km_Aggregated.tif \
  "Infaint Mortality Rate" \
  --api_key sk-xxx \
  --auxiliary_dir data \
  --output_dir ./your_output_dir \
  --model gpt-3.5-turbo-0125
```

Done!
Refined results will appear in your_output_dir.