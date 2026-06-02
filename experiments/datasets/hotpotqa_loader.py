from datasets import load_dataset

def load_hotpotqa_sample(num_samples: int = 10):
    """
    Loads HotpotQA from HuggingFace.
    We use the validation split since test is usually blind.
    """
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    return dataset.select(range(num_samples))

def format_hotpotqa_prompt(example: dict) -> str:
    """
    Formats the HotpotQA question and distractor context into a single prompt.
    """
    question = example['question']
    context = ""
    for title, sentences in zip(example['context']['title'], example['context']['sentences']):
        context += f"Title: {title}\n"
        context += " ".join(sentences) + "\n\n"
        
    prompt = f"Answer the following question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {question}"
    return prompt

def evaluate_hotpotqa_answer(predicted: str, gold: str) -> bool:
    """
    Evaluates HotpotQA answer via exact match or partial containment.
    """
    if not predicted:
        return False
    return gold.lower() in str(predicted).lower()
