import re
from typing import List
from lm_model import LMModel

def compute_selfcite_reward(lm_model: LMModel, all_context_sents: List[str], statement_text: str, citation_ids: List[int], query: str) -> float:
    Ei_list = []
    for cid in citation_ids:
        if 1 <= cid <= len(all_context_sents):
            Ei_list.append(all_context_sents[cid-1])
    Ei_text = "\n".join(Ei_list)
    CEi_list = []
    for i, s in enumerate(all_context_sents, start=1):
        if i not in citation_ids:
            CEi_list.append(s)
    CEi_text = "\n".join(CEi_list)
    prompt_Ei = Ei_text + "\n\nUser Query: " + query + "\nAnswer: "
    prompt_CEi = CEi_text + "\n\nUser Query: " + query + "\nAnswer: "
    logprob_Ei = lm_model.compute_logprob_of_output(prompt_Ei, statement_text)
    logprob_CEi = lm_model.compute_logprob_of_output(prompt_CEi, statement_text)
    reward = logprob_Ei - logprob_CEi
    return reward

def parse_citation_ids(citation_str: str) -> List[int]:
    range_pat = r"\[(\d+)-(\d+)\]"
    single_pat = r"\[(\d+)\]"
    ids = set()
    for match in re.finditer(range_pat, citation_str):
        start_id = int(match.group(1))
        end_id = int(match.group(2))
        for i in range(start_id, end_id + 1):
            ids.add(i)
    for match in re.finditer(single_pat, citation_str):
        sid = int(match.group(1))
        ids.add(sid)
    ids_list = sorted(list(ids))
    return ids_list

def sample_citation_candidates(lm_model: LMModel, partial_prompt: str, statement_text: str, n: int = 10) -> List[str]:
    base_prompt = partial_prompt + statement_text + " <cite>"
    candidates = []
    for _ in range(n):
        raw = lm_model.generate(base_prompt, max_new_tokens=30, temperature=1.2, top_p=0.9)
        m = re.search(r"<cite>(.*?)</cite>", raw, flags=re.DOTALL)
        if m:
            c_str = m.group(1).strip()
        else:
            c_str = ""
        candidates.append(c_str)
    return candidates

def best_of_n_citation(lm_model: LMModel, context_sents: List[str], query: str, statement_text: str, partial_prompt: str, n: int = 10, max_citation_length: int = 384) -> str:
    cands = sample_citation_candidates(lm_model, partial_prompt, statement_text, n=n)
    best_reward = -1e9
    best_citation_str = ""
    for c_str in cands:
        c_ids = parse_citation_ids(c_str)
        total_len = 0
        for cid in c_ids:
            if 1 <= cid <= len(context_sents):
                total_len += len(context_sents[cid-1].split())
        if total_len > max_citation_length and len(c_ids) > 1:
            continue
        reward = compute_selfcite_reward(lm_model, all_context_sents=context_sents, statement_text=statement_text, citation_ids=c_ids, query=query)
        if reward > best_reward:
            best_reward = reward
            best_citation_str = c_str
    return f"<cite>{best_citation_str}</cite>"

def generate_response_with_selfcite(lm_model: LMModel, context_sents: List[str], query: str, max_statements: int = 5) -> str:
    partial_prompt = "\n".join(context_sents) + "\n\nUser Query: " + query + "\nAnswer: "
    final_output = ""
    for _ in range(max_statements):
        statement_gen = lm_model.generate(partial_prompt, max_new_tokens=64, temperature=0.7, top_p=0.9)
        newly_added = statement_gen[len(partial_prompt):]
        if not newly_added.strip():
            break
        statement_text = newly_added.strip()
        citation_str = best_of_n_citation(lm_model, context_sents, query, statement_text, partial_prompt, n=10)
        final_output += statement_text + " " + citation_str + "\n"
        partial_prompt += statement_text + " " + citation_str
    return final_output.strip()
