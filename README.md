
<div align="center">
  <img src="https://github.com/user-attachments/assets/a2fa8cf1-86e4-4b83-a35d-1b353bc5b3a4" alt="Logo" width="150">
  <h1 align="center">PPP-Agent: Training Proactive and Personalized LLM Agents</h1>

  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![arXiv](https://img.shields.io/badge/arXiv-2511.02208-b31b1b.svg)](https://arxiv.org/abs/2511.02208)
  [![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-PPP--36B-yellow.svg)](https://huggingface.co/sunweiwei/PPP-36B)
</div>

**PPP-Agent** is an open-source framework for training LLM agents that are not only productive (task success) but also **proactive** (asking essential clarifying questions) and **personalized** (adapting to diverse user preferences). It includes **UserVille**, an interactive environment that turns existing agent benchmarks into multi-turn, preference-aware simulations.

> **[Training Proactive and Personalized LLM Agents](https://arxiv.org/abs/2511.02208)**  
> Authors: Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, Yiming Yang  
> [https://arxiv.org/pdf/2511.02208](https://arxiv.org/pdf/2511.02208)

---

## Highlights

- **UserVille**: Converts precise prompts into vague ones and simulates 20 user preferences.
- **PPP RL**: Multi-objective RL optimizing Productivity, Proactivity, and Personalization.
- **Tools**: Plug-and-play scaffolds for SWE (SWE-Bench, SWE-Gym) and Deep Research (BrowseComp+).
- **Metrics**: Effort-based proactivity and preference-following personalization.
- **Generalization**: Transfers to unseen preferences, simulators, and tasks.

---

## Model

**PPP-36B** is a Seed-36B-Instruct model trained with our PPP RL framework on SWE-Func-Loc tasks: 🤗 [sunweiwei/PPP-36B](https://huggingface.co/sunweiwei/PPP-36B)

---

## Evaluation

### 1. Start the Repo Server

Download SWE-Bench repo data to `envs/gym_data`:
```bash
cd envs
python download_swe_repo.py --dataset princeton-nlp/SWE-bench_Verified --split test
# Download SWE-Gym data: python download_swe_repo.py --dataset SWE-Gym/SWE-Gym --split train
```

Start the repo server
```bash
cd envs && python repo_server.py
```

### 2. Download Data

This contains the training and test data used in our experiments: https://drive.google.com/drive/folders/1yJHQckiRTkshF8SScZHUK3sjp9SpqW2x?usp=sharing

Place the downloaded parquet files in the `data/` directory.

### 3. Run Evaluation

Basic usage with the OpenAI API:
```bash
python scripts/eval_func_loc.py \
  --data_path data/test_ood.parquet \
  --model_name gpt-4o-mini \
  --local_repo_path /path/to/envs/gym_data \
  --local_repo_url http://localhost:8011 \
  --num_workers 1
```

Evaluate on multiple datasets:
```bash
python scripts/eval_func_loc.py \
  --data_path data/test_ood.parquet data/test_id.parquet \
  --model_name gpt-4o-mini \
  --local_repo_path /path/to/envs/gym_data \
  --local_repo_url http://localhost:8011 \
  --num_workers 1
```

**Key arguments:**
- `--local_repo_path`: Path to the gym_data directory
- `--local_repo_url`: URL of the repo server
- `--data_path`: One or more parquet files to evaluate
- `--model_name`: Model to use for evaluation

For all available arguments, run:
```bash
python scripts/eval_func_loc.py --help
```

### 4. Evaluation with PPP-36B

To evaluate with vLLM using PPP-36B or other local models:

**Note:** PPP-36B includes bias terms in attention output projections and requires a patch to serve correctly with vLLM.

```bash
# Start vLLM server with the patch (scripts/patch_seed_oss.py)
PYTHONPATH=scripts python -c "import patch_seed_oss" && vllm serve sunweiwei/PPP-36B --port 8000

# Set OPENAI_BASE_URL and run evaluation
export OPENAI_BASE_URL=http://localhost:8000/v1

python scripts/eval_func_loc.py \
  --data_path data/test_ood.parquet \
  --model_name sunweiwei/PPP-36B \
  --local_repo_path /path/to/envs/gym_data \
  --local_repo_url http://localhost:8011 \
  --num_workers 1
```

---

## Training

See `scripts/train_ppp.py` and `scripts/train_ppp.sh`

---
## Cite

If you find this work useful, please consider citing our paper:

```bibtex
@article{sun2025pppagent,
  title={Training Proactive and Personalized LLM Agents},
  author={Sun, Weiwei and Zhou, Xuhui and Du, Weihua and Wang, Xingyao and Welleck, Sean and Neubig, Graham and Sap, Maarten and Yang, Yiming},
  journal={arXiv preprint arXiv:2511.02208},
  year={2025},
  url={https://arxiv.org/abs/2511.02208}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.




