from datasets import load_dataset

def load_math_sample(num_samples: int = 10, split: str = "test"):
    """
    Loads the MATH dataset from HuggingFace.
    """
    # The 'hendrycks/competition_math' is the standard MATH dataset
    dataset = load_dataset("hendrycks/competition_math", split=split)
    
    # Optional: could filter by level or type if needed, 
    # but taking the first N samples is fine for now.
    return dataset.select(range(min(num_samples, len(dataset))))

def format_math_prompt(example: dict) -> str:
    """
    Formats the MATH problem into a prompt.
    """
    question = example['problem']
    level = example.get('level', 'Unknown')
    type_ = example.get('type', 'Unknown')
    
    prompt = f"Solve the following {level} {type_} problem step-by-step. Put your final answer at the very end.\n\nProblem: {question}"
    return prompt

def evaluate_math_answer(predicted: str, gold: str) -> bool:
    """
    Evaluates MATH dataset answer. The MATH dataset answers are typically in LaTeX.
    This is a simplified check that looks if the boxed answer is present in the text.
    For rigorous testing, you'd want to use a formal math equivalence checker.
    """
    if not predicted:
        return False
        
    # Extract just the answer string from the gold solution, which might have steps
    # Often the final answer in MATH is in a \boxed{}
    # We will just do a substring search for now as a baseline
    # A true math evaluator would strip latex formatting.
    import re
    match = re.search(r'\\boxed{(.+?)}', gold)
    if match:
        gold_ans = match.group(1).strip()
    else:
        gold_ans = gold.strip()
        
    return gold_ans in str(predicted)
