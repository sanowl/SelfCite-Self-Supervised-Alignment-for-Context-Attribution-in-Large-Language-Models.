import os
import json
import random
import argparse
import nltk

from data_utils import load_dataset, split_into_sentences, prepend_sentence_ids
from lm_model import LMModel
from simpo import train_simpo
from selfcite import generate_response_with_selfcite

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="inference", choices=["build_pref_data", "simpo", "inference"])
    parser.add_argument("--model_path", type=str, help="Hugging Face model path or identifier", required=True)
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to JSONL dataset for training or building preference data")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of preference samples to build")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate for SimPO")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha parameter in the SimPO objective")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for SimPO fine-tuning")
    args = parser.parse_args()
    stage = args.stage

    if stage == "build_pref_data":
        if not args.dataset_path:
            raise ValueError("Must provide dataset_path to build preference data.")
        print("Loading dataset...")
        raw_data = load_dataset(args.dataset_path)
        print("Loading base model...")
        lm_model = LMModel(args.model_path)
        print("Building preference data...")
        from simpo import build_preference_dataset  # local import
        pref_data = build_preference_dataset(lm_model, raw_data, num_samples=args.num_samples)
        pref_out = "pref_data.jsonl"
        with open(pref_out, 'w', encoding='utf-8') as f:
            for pd in pref_data:
                f.write(json.dumps(pd) + "\n")
        print(f"Preference data saved to {pref_out}")

    elif stage == "simpo":
        pref_data_file = "pref_data.jsonl"
        if not os.path.exists(pref_data_file):
            raise FileNotFoundError("pref_data.jsonl not found. Build it first.")
        pref_data = []
        with open(pref_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                pref_data.append(item)
        print(f"Loaded {len(pref_data)} preference pairs. Now training via SimPO.")
        train_simpo(
            base_model_path=args.model_path,
            preference_data=pref_data,
            learning_rate=args.learning_rate,
            alpha=args.alpha,
            num_epochs=args.num_epochs
        )

    elif stage == "inference":
        print("Loading model for inference...")
        lm_model = LMModel(args.model_path)
        context = "Cacti are plants that have spines... They help with water conservation..."
        query = "Why do cacti have spines?"
        sents = split_into_sentences(context)
        labeled = prepend_sentence_ids(sents)
        print("Generating with SelfCite best-of-N approach...")
        resp = generate_response_with_selfcite(lm_model, labeled, query, max_statements=3)
        print("=== Model Output ===")
        print(resp)
    else:
        raise ValueError(f"Unknown stage: {stage}")

if __name__ == "__main__":
    main()
