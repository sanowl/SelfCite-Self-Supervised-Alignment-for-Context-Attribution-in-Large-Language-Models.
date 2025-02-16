import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from typing import List, Dict
from data_utils import split_into_sentences, prepend_sentence_ids, build_model_input, load_dataset
from selfcite import generate_response_with_selfcite
import secrets

def simpo_loss_fn(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, chosen_text: str, rejected_text: str, alpha: float = 0.1) -> torch.Tensor:
    device = next(model.parameters()).device
    chosen_enc = tokenizer(chosen_text, return_tensors='pt')
    chosen_enc = {k: v.to(device) for k, v in chosen_enc.items()}
    chosen_out = model(**chosen_enc, labels=chosen_enc['input_ids'])
    chosen_len = chosen_enc['input_ids'].shape[1]
    chosen_nll_sum = chosen_out.loss * chosen_len
    rejected_enc = tokenizer(rejected_text, return_tensors='pt')
    rejected_enc = {k: v.to(device) for k, v in rejected_enc.items()}
    rejected_out = model(**rejected_enc, labels=rejected_enc['input_ids'])
    rejected_len = rejected_enc['input_ids'].shape[1]
    rejected_nll_sum = rejected_out.loss * rejected_len
    diff_log_p = -(chosen_nll_sum - rejected_nll_sum)
    loss = alpha * (-diff_log_p) + chosen_out.loss + rejected_out.loss
    return loss

def build_preference_dataset(lm_model, raw_data: List[Dict], num_samples: int = 2000) -> List[Dict]:
    pref_data = []
    subset = raw_data[:num_samples]
    for ex in subset:
        context = ex['context']
        query = ex['query']
        sents = split_into_sentences(context)
        labeled_sents = prepend_sentence_ids(sents)
        partial_prompt = build_model_input(labeled_sents, query)
        original_resp = lm_model.generate(partial_prompt, max_new_tokens=256, temperature=0.7, top_p=0.9)
        improved_resp = generate_response_with_selfcite(lm_model, labeled_sents, query)
        item = {"context_sents": labeled_sents, "query": query, "chosen_response": improved_resp, "rejected_response": original_resp}
        pref_data.append(item)
    return pref_data

def train_simpo(base_model_path: str, preference_data: List[Dict], learning_rate=5e-7, alpha=0.1, num_epochs=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(preference_data) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=total_steps)
    idx_list = list(range(len(preference_data)))
    step_count = 0
    for epoch in range(num_epochs):
        secrets.SystemRandom().shuffle(idx_list)
        for i in idx_list:
            item = preference_data[i]
            chosen_text = item['chosen_response']
            rejected_text = item['rejected_response']
            optimizer.zero_grad()
            loss = simpo_loss_fn(model, tokenizer, chosen_text, rejected_text, alpha=alpha)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step_count += 1
            if step_count % 50 == 0:
                print(f"[Epoch {epoch}] Step {step_count} loss={loss.item():.4f}")
    out_dir = "simpo_selfcite_model"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"SimPO training complete. Model saved at {out_dir}")
    return out_dir
