
<div align="center">
  <img src="https://github.com/user-attachments/assets/a2fa8cf1-86e4-4b83-a35d-1b353bc5b3a4" alt="Logo" width="150">
  <h1 align="center">PPP-Agent: Training Proactive and Personalized LLM Agents</h1>

  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![arXiv](https://img.shields.io/badge/arXiv-2511.02208-b31b1b.svg)](https://arxiv.org/abs/2511.02208)
</div>

**PPP-Agent** is an open-source framework for training LLM agents that are not only productive (task success) but also **proactive** (ask essential clarifying questions) and **personalized** (adapt to diverse user preferences). It includes **UserVille**, an interactive environment that turns existing agent benchmarks into multi-turn, preference-aware simulations.

> **[Training Proactive and Personalized LLM Agents](https://arxiv.org/abs/2511.02208)**  
> Author: Weiwei Sun, Xuhui Zhou, Weihua Du, Xingyao Wang, Sean Welleck, Graham Neubig, Maarten Sap, Yiming Yang  
> [https://arxiv.org/pdf/2511.02208](https://arxiv.org/pdf/2511.02208)

---

## Highlights

- **UserVille**: converts precise prompts into **vague** ones and simulates users with 20 configurable preferences.
- **PPP RL**: multi-objective RL optimizing **Productivity, Proactivity, Personalization** jointly.
- **Plug-and-Play Tools**: SWE (SWE-Bench & SWE-Gym) and Deep-Research (BrowseComp+) scaffolds.
- **User-Centric Metrics**: effort-based proactivity + preference-following personalization.
- **Generalization**: transfers to unseen preferences, simulators, and downstream tasks.

---

## Model

We release **PPP-36B**, a Seed-36B-Instruct model trained with our PPP RL framework: 🤗 [sunweiwei/PPP-36B](https://huggingface.co/sunweiwei/PPP-36B)

---

## Evaluation

**1. Start Repo Server**

Download SWE-Bench repo data to `envs/gym_data`:
```bash
cd envs
python download_swe_repo.py --dataset princeton-nlp/SWE-bench_Verified --split test
# download swe-gym data: python download_swe_repo.py --dataset SWE-Gym/SWE-Gym --split train
```

Start the repo server
```bash
cd envs && python repo_server.py
```

**2. Download Data**

This contain the training and test data used in our experiment: https://drive.google.com/drive/folders/1yJHQckiRTkshF8SScZHUK3sjp9SpqW2x?usp=drive_link

Place the downloaded parquet files in the `data/` directory.

**3. Run Evaluation**

Basic usage with OpenAI API:
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

**4. Evaluation with PPP-36B**

To evaluate using vLLM with PPP-36B or other local models:

**Note:** PPP-36B includes bias terms in attention output projections and requires a patch to serve correctly with vLLM.

```bash
# Start vLLM server with the patch (included in scripts/)
PYTHONPATH=scripts python -c "import patch_seed_oss" && vllm serve sunweiwei/PPP-36B --port 8000

# In a new terminal, set environment variables and run evaluation
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy  # vLLM doesn't require a real key

python scripts/eval_func_loc.py \
  --data_path data/test_ood.parquet \
  --model_name sunweiwei/PPP-36B \
  --local_repo_path /path/to/envs/gym_data \
  --local_repo_url http://localhost:8011 \
  --num_workers 1
```

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






