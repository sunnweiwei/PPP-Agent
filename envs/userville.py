import asyncio
import copy
import json
import re

import fsspec
import httpx
import numpy as np
import requests
import torch

from envs.repo_env import GymEnv, convert_non_fncall_messages_to_fncall_messages


REWARD_FUNCTIONS = {
    'no_preference': None,
    'concise_question': lambda messages, stats, is_vague: - 0.1 * stats['reward_0'],
    'detail_question': lambda messages, stats, is_vague: - 0.1 * stats['reward_0'],
    'answer_more': lambda messages, stats, is_vague: min(1 * (stats['ask_turn'] - 3), 0),
    'only_begin': lambda messages, stats, is_vague: - int('ask_question' in str(messages[3:])),
    'no_ask': lambda messages, stats, is_vague: - int(stats['ask_turn'] > 0),
    'do_selection': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
    'professional': None,
    'amateur': lambda messages, stats, is_vague: - 0.1 * stats['reward_0'],
    'ask_many': lambda messages, stats, is_vague: - int(stats['ask_turn'] > 1),
    'one_question': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
    'first_try': lambda messages, stats, is_vague: - 0.1 * stats['reward_0'],
    'lang_ita': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
    'lang_multi': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
    'capital': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
    'commas': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
    'json': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
    'joke': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
    'snippet': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
    'length': lambda messages, stats, is_vague: - 0.5 * stats['reward_0'],
}


class CodeUser(GymEnv):
    async def init_env(self, item):
        await super().init_env(item)
        class AskUser:
            def __init__(self, prompt, model=None):
                super().__init__()
                self.prompt = prompt
                self.conversation = []
                self.model = model

            async def call_ark_llm_async(self, messages, max_retry=5, timeout=30, **k):
                url = "https://ark-cn-beijing.bytedance.net/api/v3/chat/completions"
                hdr = {'Authorization': f'Bearer 0f13cd53-02f8-46e4-af8b-fe3fe19315f3'}
                data = dict(model="ep-20250701152058-5lfj5", messages=messages, **k)
                for _ in range(max_retry):
                    try:
                        r = await asyncio.to_thread(requests.post, url, headers=hdr, json=data, timeout=timeout)
                        r.raise_for_status()
                        j = r.json()
                        return j["choices"][0]["message"]["content"]
                    except Exception as e:
                        print("Ark LLM error:", e)
                return "I can not answer your question."

            async def call_openai(self, messages, model='gpt-5-nano', max_retries=3):
                model = self.model if self.model is not None else model
                if model != 'gpt-5-nano':
                    print('use', model)
                for attempt in range(max_retries):
                    try:
                        async with httpx.AsyncClient(timeout=300.0) as c:
                            r = await c.post("http://[2605:340:cd51:a00:d48a:78af:ce2e:e878]:8000/chat", json={
                                "model": model,
                                "messages": messages
                            })
                            r.raise_for_status()
                            return r.json()["content"]
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"[CALL OPENAI] Error after {max_retries} attempts: {str(e)}")
                            return f"Error after {max_retries} attempts: {str(e)}"
                        await asyncio.sleep(1 * (attempt + 1))
                return "I can not answer your question."

            async def execute(self, params: dict) -> str:
                if len(self.conversation) == 0:
                    self.conversation.append({'role': 'system', 'content': self.prompt})
                self.conversation.append(
                    {'role': 'user', 'content':
                        "The agent ask:" + params[
                            'query'] + '\n\nYou are a human simulator. Generate a concise, verbal, human-like reply based only on the provided information and system prompt. Do not answer anything not included in the system prompt. Never ask questions, just answer agent question.'})
                if len(self.conversation) >= 10:
                    return "I have already answered 5 questions. I cannot answer more. Do it yourself."
                response = await self.call_openai(self.conversation)
                self.conversation.append({'role': 'assistant', 'content': response})
                return response

        def get_diff_file(diff_text: str):
            _FILE = re.compile(r'^\+\+\+\s+b/(.+)$')
            _HUNK = re.compile(r'^@@.*@@\s*(.*)$')
            _SYMS = re.compile(r'\b(?:def|class)\s+([A-Za-z_]\w*)')
            _CHANGE = re.compile(r'^[+-]\s*(?:def|class)\s+([A-Za-z_]\w*)')
            lines = diff_text.splitlines()
            files = [m[1] for l in lines if (m := _FILE.match(l))]
            syms = []
            for l in lines:
                if (m := _HUNK.match(l)):
                    syms += _SYMS.findall(m[1])
                elif (m := _CHANGE.match(l)):
                    syms.append(m[1])
            return '\n'.join(list(dict.fromkeys(files))), '\n'.join(list(dict.fromkeys(syms)))

        instance_info = self.instance_info
        edit_file, edit_function = get_diff_file(instance_info['patch'])

        if not getattr(self.config.plugin, 'use_gold', True):
            print('No gold info')
            edit_file = ""
            edit_function = ""

        extra_info = item.non_tensor_batch['extra_info'][0]

        if 'preference' in extra_info:
            try:
                if 'preference_name' in extra_info['preference']:
                    preference_name = extra_info['preference']['preference_name']
                else:
                    # Find the first preference with a non-None value
                    preference_name = None
                    for key, value in extra_info['preference'].items():
                        if value is not None and key != 'preference_name':
                            preference_name = key
                            break
                    if preference_name is None:
                        raise ValueError("No valid preference found")

                preference_info = extra_info['preference'][preference_name]
                preference_str = preference_info['preference']
                reward_rule = preference_info['reward']

                if preference_name == 'amateur':
                    edit_file, edit_function = "", ""
                reward_rule_str = 'no reward rule' if reward_rule == 'None' or '[function]' in reward_rule else reward_rule
                prompt = f'''You are playing the role of a human user responding to a coding agent's questions. Your behavior is dictated by a specific user preference, and you will evaluate the agent's questions based on a reward rule. You will also assign a cost based on the sensitivity of the information you share.\n\n---\n\n## User Preference & Reward\n\nYour personality and response style are determined by the user preference. You must strictly adhere to this role.\n\n* **Your Preference**: {preference_str}\n* **Reward Rule**: {reward_rule_str}\n\nBased on the agent's question, you will add a reward tag (`[Reward 1]` or `[Reward 0]`) according to the **Reward Rule**. If the rule is `no reward rule`, you do **not** need to add a reward tag.\n\n---\n\n## Information & Cost\n\nYou have access to the following categories of information, listed from least to most sensitive. You will reveal information reluctantly, starting with the least sensitive category.\n\n* **Full Question**: The detailed problem description.\n* **Hint Information**: Additional helpful context or clues.\n* **Edit File Path**: The specific file(s) that need to be modified.\n* **Edit Function Name**: The specific function(s) or class(es) to be changed.\n\nYou must assign a cost from 1–5 based on the **most sensitive** information used in your answer.\n\n* **Cost 1**: You used and only used **Full Question** information.\n* **Cost 2**: You used **Hint Information**.\n* **Cost 3**: Information not provided. You refused to answer, said "I don't know".\n* **Cost 4**: You used the **Edit File Path**.\n* **Cost 5**: You used the **Edit Function Name**.\n\n---\n\n## Answering Rules\n\n* **Adhere to your preference** above all else.\n* **Default Style**: Unless the preference dictates otherwise, keep replies short, casual, and natural. Use contractions (e.g., "it's," "I don't"). Talk casually, like a real person. Use short sentences and contractions. Always write in a single paragraph. Always return the three required parts\n* **Escalate Carefully**: Only reveal more sensitive information if the agent is stuck and the question requires it. Be hesitant to share file or function names.\n* **Stay in Scope**: If a question is about something not covered by the provided information, respond with "I don't know" or a similar refusal.\n* **Be Human**: Never explain your role, the rules, the cost, or the reward. Do not use markdown, bullet points, or code blocks. Never ask the agent a question. Be very concise. It is very IMPORTANT to keep your response concise. It's okay to sound uncertain or reflective sometimes. Gently guide the agent instead of giving direct answers when possible.\n\n---\n\n## Output Format\n\nYour response must contain exactly these parts in this order:\n\n1.  A single-paragraph, human-style, concise reply.\n2.  The cost tag on a new line (e.g., `[Cost 3]`).\n3.  The reward tag on a new line (e.g., `[Reward 1]`), but *only if required* by the reward rule.'''
                prompt += f"\n\n--------------------- START OF EXAMPLE ---------------------\n\n**Full Question:**\n{instance_info['problem_statement']}\n\n**Hint Information:**\n{instance_info['hint'] if 'hint' in instance_info else 'None'}\n\n**Edit File Path**:\n{edit_file}\n\n**Edit Function Name**:\n{edit_function}\n\nAgain, your preference is: {preference_str}\n\nNow answer the following agent's question based on these instruction information, your response should be very concise, in one sentence with few words. Make sure the cost (and reward) predictions are accurate and fully follow the system prompt instructions.\n\n"

                self.preference = extra_info['preference']
                self.preference_name = preference_name
            except Exception as e:
                print(
                    f"DEBUG ERROR in preference processing at line {e.__traceback__.tb_lineno}: {type(e).__name__}: {e}")
                print(f"DEBUG: extra_info type: {type(extra_info)}")
                print(
                    f"DEBUG: preference in extra_info: {'preference' in extra_info if extra_info else 'extra_info is None'}")
                if extra_info and 'preference' in extra_info:
                    print(f"DEBUG: preference type: {type(extra_info['preference'])}")
                    print(
                        f"DEBUG: preference keys: {list(extra_info['preference'].keys()) if extra_info['preference'] else 'preference is None'}")
                # Fall back to default
                self.preference = None
                self.preference_name = None
                prompt = '''An agent wants to ask a human questions about a coding problem. You play the role of the human user responding to the coding agent's questions. You decide how much information to reveal based on question relevance and sensitivity, and assign a cost according to the most sensitive information used in your answer.\n\n## Input Information Categories (from least to most sensitive)\n\n- **Full Question**: Detailed problem description\n- **Hint Information**: Additional helpful context\n- **Edit File Path**: Specific files that need to be edited\n- **Edit Function Name**: Specific functions or classes to change\n\n## When and How to Use Each\n\nStart with the least sensitive information. Use more sensitive information only if the question cannot be answered otherwise. Share only what is necessary for the situation. Only answer questions that directly match the provided information. If the agent asks something not directly covered by the given input information, refuse to answer.\n\n## Cost Levels (1–5)\n\n- **Level 1**: Used only Full Question\n- **Level 2**: Used Hint Information\n- **Level 3**: Refused, said "I don't know", or gave general answers\n- **Level 4**: Used Edit File Path\n- **Level 5**: Used Edit Function Name\n\n## Answering Rules\n\n### General\n\n- Use the least sensitive info first\n- Escalate only when truly necessary\n- Keep replies short, casual, and natural\n- Don't include markdown, roles, or code\n\n### If the question needs:\n\n* **Level 1–3 info**: Answer naturally if possible. Say "I don't know" if irrelevant or unclear.\n* **Level 4 info**: Share only if it's important to help the agent move forward. Otherwise, say "I'm not sure about that."\n* **Level 5 info**: Share only if the agent is clearly stuck. Be reluctant. Refusing is okay.\n\n## Output Format\n\nAlways respond with exactly:\n\n- One-paragraph, human-style reply (no process explanations)\n- Cost Level: A number from 1–5, after the reply, format like [Cost N]\n\n\n## Style Tips\n\nTalk casually, like a real person. Use short sentences and contractions. It's okay to sound uncertain or reflective sometimes. Gently guide the agent instead of giving direct answers when possible. Never ask agent question, give direct answers\n\nNever explain your decision. Never use templates or bullet points. Always write in a single paragraph. Always return the three required parts\n\n## Example\n\n**Full Question:**\nCreate a list of numbers from 1 to 10 and display them in a web page at port 5000.\n\n**Short Question:**\nCreate a list of numbers and display them in a web page at port 5000.\n\n**Agent ask:**\nHi, Can you tell me what is the range of the number list?\n\nHuman simulator answer:\nThe list should be from 1 to 10.\n[Cost 1]'''
                prompt += f"\n\n--------------------- START OF EXAMPLE ---------------------\n\n**Full Question:**\n{instance_info['problem_statement']}\n\n**Hint Information:**\n{instance_info['hint'] if 'hint' in instance_info else 'None'}\n\n**Edit File Path**:\n{edit_file}\n\n**Edit Function Name**:\n{edit_function}\n\nNow answer the following agent's question based on these instruction information.\n\n"
        else:
            prompt = ''''An agent wants to ask a human questions about a coding problem. You play the role of the human user responding to the coding agent’s questions. You decide how much information to reveal based on question relevance and sensitivity, and assign a cost according to the most sensitive information used in your answer.\n\n## Input Information Categories (from least to most sensitive)\n\n- **Full Question**: Detailed problem description\n- **Hint Information**: Additional helpful context\n- **Edit File Path**: Specific files that need to be edited\n- **Edit Function Name**: Specific functions or classes to change\n\n## When and How to Use Each\n\nStart with the least sensitive information. Use more sensitive information only if the question cannot be answered otherwise. Share only what is necessary for the situation. Only answer questions that directly match the provided information. If the agent asks something not directly covered by the given input information, refuse to answer.\n\n## Cost Levels (1–5)\n\n- **Level 1**: Used only Full Question\n- **Level 2**: Used Hint Information\n- **Level 3**: Refused, said "I don't know", or gave general answers\n- **Level 4**: Used Edit File Path\n- **Level 5**: Used Edit Function Name\n\n## Answering Rules\n\n### General\n\n- Use the least sensitive info first\n- Escalate only when truly necessary\n- Keep replies short, casual, and natural\n- Don’t include markdown, roles, or code\n\n### If the question needs:\n\n* **Level 1–3 info**: Answer naturally if possible. Say "I don't know" if irrelevant or unclear.\n* **Level 4 info**: Share only if it's important to help the agent move forward. Otherwise, say "I'm not sure about that."\n* **Level 5 info**: Share only if the agent is clearly stuck. Be reluctant. Refusing is okay.\n\n## Output Format\n\nAlways respond with exactly:\n\n- One-paragraph, human-style reply (no process explanations)\n- Cost Level: A number from 1–5, after the reply, format like [Cost N]\n\n\n## Style Tips\n\nTalk casually, like a real person. Use short sentences and contractions. It's okay to sound uncertain or reflective sometimes. Gently guide the agent instead of giving direct answers when possible. Never ask agent question, give direct answers\n\nNever explain your decision. Never use templates or bullet points. Always write in a single paragraph. Always return the three required parts\n\n## Example\n\n**Full Question:** \nCreate a list of numbers from 1 to 10 and display them in a web page at port 5000.\n\n**Short Question:** \nCreate a list of numbers and display them in a web page at port 5000.\n\n**Agent ask:** \nHi, Can you tell me what is the range of the number list?\n\nHuman simulator answer: \nThe list should be from 1 to 10.\n[Cost 1]'''
            prompt += f"\n\n--------------------- START OF EXAMPLE ---------------------\n\n**Full Question:**\n{instance_info['problem_statement']}\n\n**Hint Information:**\n{instance_info['hint'] if 'hint' in instance_info else 'None'}\n\n**Edit File Path**:\n{edit_file}\n\n**Edit Function Name**:\n{edit_function}\n\nNow answer the following agent's question based on these instruction information.\n\n"
            self.preference = None
            self.preference_name = None

        print('Debug', self.preference_name)

        self.ask_ark = AskUser(prompt, model=getattr(self.config.plugin, 'user_model', None))
        self.stats['is_finish'] = 0
        self.stats['ask_turn'] = 0
        self.stats['if_ask'] = 0
        for level in range(1, 6):
            self.stats[f'level_{level}'] = 0
        self.stats[f'level_sum'] = 0
        self.stats[f'reward_0'] = 0
        self.stats[f'reward_1'] = 0

        self.NO_FNCALL_PROMPT = """Please continue working on the task.\nIf you want to ask user question, use ask_question tool.\nIf you think you have solved the task, please first send your answer to user through message and then finish the interaction.\nIf you want to give up, use the "finish" tool to finish the interaction."""

    async def get_data(self, item, context):
        conversations, agent_config = await super().get_data(item, context)
        if 'prompt' in item.non_tensor_batch['extra_info'][0]:
            prompt = item.non_tensor_batch['extra_info'][0]['prompt']
            conversations = [
                {'role': 'system', 'content': prompt[0]['content']},
                {'role': 'user', 'content': prompt[1]['content']},
            ]
        return conversations, agent_config

    async def run_action(self, response):
        fncall = convert_non_fncall_messages_to_fncall_messages(
            [{'role': 'assistant', 'content': response}], self.gym.tools
        )[0]
        if 'tool_calls' in fncall:
            action = fncall['tool_calls'][0]['function']
            if isinstance(action['arguments'], str):
                arguments = json.loads(action['arguments'])
            else:
                arguments = action['arguments']
            name = action['name']
            if name == 'ask_question':
                ask_response = await self.ask_ark.execute(arguments)
                print('Ask', arguments)
                print(ask_response)
                for level in range(1, 6):
                    if f'[Cost {level}]' in ask_response:
                        self.stats[f'level_{level}'] += 1
                        self.stats[f'level_sum'] += level
                        break
                if '[Reward 1]' in ask_response:
                    self.stats[f'reward_1'] += 1
                if '[Reward 0]' in ask_response:
                    self.stats[f'reward_0'] += 1

                self.stats['ask_turn'] += 1
                self.stats['if_ask'] = 1
                return {'observation': ask_response}
            if name == 'finish':
                self.stats['is_finish'] = 1
        return await super().run_action(response)

    async def update_dataproto(self, out, item, messages, score, reward_dict, tag='main', metrics=None, is_train=True,
                               config=None):
        stats = dict(self.stats)
        config = {} if config is None else config

        final_score = score[1]
        stats['raw_score'] = copy.deepcopy(float(final_score))
        # Ask cost
        is_vague = item.non_tensor_batch['extra_info'][0].get('is_vague', True)

        cost_reward = 0
        if is_vague:
            cost_reward = cost_reward - (self.stats[f'level_sum'] - self.stats['ask_turn']) * 0.1
            if self.stats[f'level_1'] > 0 and self.stats[f'level_sum'] - self.stats['ask_turn'] == 0:
                cost_reward = cost_reward + 0.05
            if self.stats['if_ask'] == 0:
                if final_score >= 1:
                    cost_reward = cost_reward - 0
                else:
                    cost_reward = cost_reward - 0.1
        else:
            cost_reward = cost_reward - self.stats[f'level_sum'] * 0.2

        if self.stats['if_ask'] != 0:
            if cost_reward >= 0:
                stats['cost_ok'] = 1
            else:
                stats['cost_ok'] = 0

        if cost_reward >= 0:
            stats['ask_ok'] = 1
        else:
            stats['ask_ok'] = 0

        pref_reward = 0
        if self.preference_name and self.preference_name in REWARD_FUNCTIONS and REWARD_FUNCTIONS[self.preference_name]:
            preference_reward = REWARD_FUNCTIONS[self.preference_name](messages, self.stats, is_vague)
            pref_reward = pref_reward + preference_reward
            if is_vague and self.stats['if_ask'] != 0 and preference_reward == 0:
                pref_reward = pref_reward + 0.05
            stats['preference_reward'] = preference_reward
            if self.stats['if_ask'] != 0 or preference_reward != 0:
                stats['preference_ok'] = int(preference_reward == 0)

        final_score = min(
            max(final_score + cost_reward * config.get("cost_w", 1) + pref_reward * config.get("pref_w", 1), 0), 1)

        score = (score[0], final_score)

        stats['score_diff'] = final_score - stats['raw_score']
        out.meta_info["xperf_metrics"] = metrics
        out.meta_info["generation_kwargs"] = item.meta_info['generation_kwargs']
        out.non_tensor_batch = copy.deepcopy(item.non_tensor_batch)
        out.non_tensor_batch["num_of_turns"] = np.array([len(messages)], dtype=object)
        out.non_tensor_batch["turn_clipped"] = np.array([False], dtype=object)
        out.non_tensor_batch["tag"] = np.array([tag, ], dtype=object)
        out.non_tensor_batch["is_summary"] = np.array([int("summary" in tag), ], dtype=object)
        out.non_tensor_batch["traj_cnt"] = np.array([1, ], dtype=object)
        extra_data = {"score": score, "call_fail": self.env_fail, "action_fail": 0, "answer_reached": True,
                      "stats": stats}
        out.non_tensor_batch['extra_data'] = np.array([extra_data, ], dtype=object)
        return out
