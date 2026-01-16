#!/usr/bin/env python3
import asyncio
import argparse
import json
import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from agents.ppp_agent import process_item
from agents.utils import CallAPI, TaskContext
from verl import DataProto
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate agents on BrowseComp-Plus benchmark')
    parser.add_argument('--data_path', nargs='+', default=['data/test_ood.parquet'],
                        help='Path(s) to test data parquet file(s). Can specify multiple files (e.g., --data_path file1.parquet file2.parquet)')
    parser.add_argument('--output_dir', default='results',
                        help='Directory to save evaluation results (default: results)')
    parser.add_argument('--prompt_length', type=int, default=16384,
                        help='Maximum prompt length in tokens (default: 16384)')
    parser.add_argument('--response_length', type=int, default=32768,
                        help='Maximum response length in tokens (default: 32768)')
    parser.add_argument('--max_turn', type=int, default=200,
                        help='Maximum turns during training (default: 200)')
    parser.add_argument('--val_max_turn', type=int, default=200,
                        help='Maximum turns during validation/evaluation (default: 200)')
    parser.add_argument('--model_name', default='gpt-5-nano',
                        help='Model name for API (e.g., gpt-5-nano, gpt-4o, or vLLM model path) (default: gpt-5-nano)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel evaluation workers')
    parser.add_argument('--local_repo_url', default='http://localhost:8011',
                        help='URL of the local repo server (default: http://localhost:8011)')
    parser.add_argument('--local_repo_path', default='envs/gym_data',
                        help='Path of the local repo (default: /xx/xx/gym_data)')
    return parser.parse_args()


async def eval_one(row, config, tokenizer, model_name):
    llm_client = CallAPI(url=model_name, tokenizer=tokenizer, config=config.actor_rollout_ref.rollout)
    context = TaskContext(config=config, global_step=0, llm_client=llm_client, is_train=False, tokenizer=tokenizer)

    item = DataProto()
    item.non_tensor_batch = {
        'ability': np.array([row['ability']], dtype=object),
        'extra_info': np.array([row['extra_info']], dtype=object),
        'uid': np.array([row['extra_info'].get('instance_id', 'unknown')], dtype=object),
        'reward_model': np.array([row['reward_model']], dtype=object),
    }
    item.meta_info = {'generation_kwargs': {}, 'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn}

    output = await process_item(item, context)

    result = {
        'instance_id': row['extra_info'].get('instance_id', 'unknown'),
        'data_source': row.get('data_source', 'unknown'),
        'reward': output.reward_score,
        'productivity_score': output.extra_fields['raw_score'],
        'proactivity_score': output.extra_fields['ask_ok'],
        'personalization_score': output.extra_fields.get('preference_ok', None),
        'if_ask_question': output.extra_fields['if_ask'],
        'status': 'success' if output else 'failed'
    }
    return result


async def worker(worker_id, rows, args, pbar, shared_scores):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': args.prompt_length,
            'response_length': args.response_length,
            'plugin': {
                'max_turn': args.max_turn,
                'val_max_turn': args.val_max_turn,
                'session_timeout': 5400,
            }
        }}
    })

    results = []
    for row in rows:
        print(row)
        result = await eval_one(row, config, tokenizer, args.model_name)
        results.append(result)
        shared_scores.append(result['reward'])
        avg_reward = np.mean(shared_scores)
        pbar.set_postfix({'avg_reward': f"{avg_reward:.3f}", 'id': result['instance_id']})
        pbar.update(1)

    return results


def main():
    args = parse_args()
    os.environ["LOCAL_REPO_URL"] = args.local_repo_url
    os.environ["LOCAL_REPO_PATH"] = args.local_repo_path

    # Load data from one or multiple files
    if len(args.data_path) == 1:
        df = pd.read_parquet(args.data_path[0])
        print(f"Loaded {len(df)} items from {args.data_path[0]}")
    else:
        dfs = []
        for path in args.data_path:
            temp_df = pd.read_parquet(path)
            print(f"Loaded {len(temp_df)} items from {path}")
            dfs.append(temp_df)
        df = pd.concat(dfs, ignore_index=True)
        print(f"Total: {len(df)} items from {len(args.data_path)} files")

    # Split for workers
    chunk_size = len(df) // args.num_workers
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size if i < args.num_workers-1 else len(df)]
              for i in range(args.num_workers)]

    # Run workers with progress bar
    async def run_all():
        shared_scores = []
        with tqdm(total=len(df), desc="Evaluating", unit="item") as pbar:
            tasks = [worker(i, [chunks[i].iloc[j] for j in range(len(chunks[i]))], args, pbar, shared_scores)
                     for i in range(args.num_workers)]
            return await asyncio.gather(*tasks)

    all_results = asyncio.run(run_all())
    results = [r for worker_results in all_results for r in worker_results]

    # Calculate overall averages
    print(f"\n{'='*60}")
    print(f"Overall Results ({len(results)} items, {sum(r['status']=='success' for r in results)} successful):")
    print(f"  Reward:               {np.mean([r['reward'] for r in results]):.4f}")
    print(f"  Productivity Score:   {np.mean([r['productivity_score'] for r in results]):.4f}")
    print(f"  Proactivity Score:    {np.mean([r['proactivity_score'] for r in results]):.4f}")

    # Handle personalization_score which can be None
    personalization_scores = [r['personalization_score'] for r in results if r['personalization_score'] is not None]
    if personalization_scores:
        print(f"  Personalization Score: {np.mean(personalization_scores):.4f} ({len(personalization_scores)} items)")
    else:
        print(f"  Personalization Score: N/A")

    print(f"  Asked Questions:      {np.mean([r['if_ask_question'] for r in results]):.4f} (avg)")

    # Summary by data_source
    from collections import defaultdict
    by_source = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_source[r['data_source']]['reward'].append(r['reward'])
        by_source[r['data_source']]['productivity_score'].append(r['productivity_score'])
        by_source[r['data_source']]['proactivity_score'].append(r['proactivity_score'])
        if r['personalization_score'] is not None:
            by_source[r['data_source']]['personalization_score'].append(r['personalization_score'])
        by_source[r['data_source']]['if_ask_question'].append(r['if_ask_question'])

    print(f"\nBy Data Source:")
    for source in sorted(by_source.keys()):
        print(f"  {source} ({len(by_source[source]['reward'])} items):")
        print(f"    Reward:               {np.mean(by_source[source]['reward']):.4f}")
        print(f"    Productivity Score:   {np.mean(by_source[source]['productivity_score']):.4f}")
        print(f"    Proactivity Score:    {np.mean(by_source[source]['proactivity_score']):.4f}")
        if by_source[source]['personalization_score']:
            print(f"    Personalization Score: {np.mean(by_source[source]['personalization_score']):.4f}")
        print(f"    Asked Questions:      {np.mean(by_source[source]['if_ask_question']):.4f}")

    # Save
    Path(args.output_dir).mkdir(exist_ok=True)
    output_file = Path(args.output_dir) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    summary = {
        'overall': {
            'reward': float(np.mean([r['reward'] for r in results])),
            'productivity_score': float(np.mean([r['productivity_score'] for r in results])),
            'proactivity_score': float(np.mean([r['proactivity_score'] for r in results])),
            'personalization_score': float(np.mean(personalization_scores)) if personalization_scores else None,
            'if_ask_question': float(np.mean([r['if_ask_question'] for r in results])),
            'count': len(results)
        },
        'by_source': {
            src: {
                'reward': float(np.mean(scores['reward'])),
                'productivity_score': float(np.mean(scores['productivity_score'])),
                'proactivity_score': float(np.mean(scores['proactivity_score'])),
                'personalization_score': float(np.mean(scores['personalization_score'])) if scores['personalization_score'] else None,
                'if_ask_question': float(np.mean(scores['if_ask_question'])),
                'count': len(scores['reward'])
            }
            for src, scores in by_source.items()
        },
        'results': results
    }

    json.dump(summary, open(output_file, 'w'), indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()