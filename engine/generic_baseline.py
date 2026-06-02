import json
from typing import List, Dict, Any, Optional
from RLM.utils.llm import LLMClient
from RLM.repl import REPLEnv
from RLM.utils import utils
from RLM.utils.tracing import TraceStorage

class GenericToolBaseline:
    """
    Standard ReAct baseline. 
    Uses tool calls (search, calc) but lacks recursive engine, 
    persistent memory, and adaptive compute depth.
    """
    def __init__(self, model: str = "ollama/llama3.1:8b", max_steps: int = 5):
        self.llm = LLMClient(model=model)
        self.max_steps = max_steps
        self.tracer = TraceStorage()
        
    def completion(self, query: str, context: str) -> str:
        self.tracer.reset()
        self.tracer.set_query(query)
        
        env = REPLEnv()
        env.load_context(context)
        
        prompt = f"""You are a helpful assistant with access to tools. Solve the query using the provided context.
Available Tools:
- search_context("query"): Finding facts in the context.
- print(val): Inspect values.

Query: {query}

Use ```repl ... ``` for tool calls. When done, write FINAL(answer)."""

        messages = [{"role": "system", "content": prompt}]
        
        for i in range(self.max_steps):
            response = self.llm.completion(messages)
            code_blocks = utils.find_code_blocks(response)
            
            if not code_blocks:
                # If no code but FINAL(answer) in text
                ans = utils.find_final_answer(response)
                if ans:
                    self.tracer.set_predicted_answer(ans[0])
                    return ans[0]
                break
                
            # Execute code
            res = env.execute(code_blocks[0])
            self.tracer.add_repl_step(
                iteration=i+1,
                response=response,
                code=code_blocks[0],
                stdout=res["stdout"],
                stderr=res["stderr"]
            )
            
            # Feedback to LLM
            exec_msg = f"Stdout: {res['stdout']}\nStderr: {res['stderr']}"
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": exec_msg})
            
        # Extraction fallback
        extract_prompt = (
            "Based on ALL research and code outputs above, give the FINAL ANSWER to: "
            f'"{query}"\n\n'
            "Rules:\n"
            "- Output ONLY the answer (a name, date, yes/no, or very short phrase).\n"
            "- Do NOT write code, sentences, or explanations.\n"
            "- Do NOT use FINAL() here - just write the raw answer.\n"
            "- Maximum 10 words.\n\nAnswer:"
        )
        messages.append({"role": "user", "content": extract_prompt})
        raw = self.llm.completion(messages).strip()
        
        import re
        raw = raw.replace("```repl", "").replace("```python", "").replace("```", "").strip()
        m = re.search(r'FINAL(?:_VAR)?\s*\(([^)]+)\)', raw)
        if m:
            raw = m.group(1).strip("'\" ")
            
        if len(raw.split()) > 15:
            first_sentence = re.split(r'[.!?\n]', raw)[0].strip()
            if first_sentence:
                raw = first_sentence
                
        return raw or "unknown"
