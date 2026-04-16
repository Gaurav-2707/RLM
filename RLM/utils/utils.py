import re
from typing import Optional,List,Dict,Tuple,Any

def find_code_blocks(text:str) -> List[str]:
    pattern = r'```repl\s*\n(.*?)\n```'
    results = []

    for match in re.finditer(pattern,text,re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)

    return results

def find_final_answer(text:str) -> Optional[Tuple[str,str]]:
    """Parse FINAL_VAR(name) or FINAL(answer) from model output.
    
    Ignores FINAL() calls that appear inside ```repl``` code blocks.
    """
    # Strip out code blocks so we don't match FINAL() inside them
    cleaned = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Try FINAL_VAR first
    match = re.search(r'FINAL_VAR\s*\(([^)]+)\)', cleaned)
    if match:
        return ('FINAL_VAR', match.group(1).strip())
    
    # Try FINAL(...)  — use greedy match but only on the cleaned text
    match = re.search(r'FINAL\s*\((.+)\)\s*$', cleaned, re.DOTALL | re.MULTILINE)
    if match:
        content = match.group(1).strip()
        # Strip wrapping quotes if the model wrote FINAL("answer")
        if (content.startswith('"') and content.endswith('"')) or \
           (content.startswith("'") and content.endswith("'")):
            content = content[1:-1]
        return ('FINAL', content)
    
    # Fallback: looser match for FINAL(text) not at end of line
    match = re.search(r'FINAL\s*\(([^)]+)\)', cleaned)
    if match:
        content = match.group(1).strip()
        if (content.startswith('"') and content.endswith('"')) or \
           (content.startswith("'") and content.endswith("'")):
            content = content[1:-1]
        return ('FINAL', content)
    
    return None


def _looks_like_code(text: str) -> bool:
    """Heuristic: does this string look like Python code rather than a plain-text answer?"""
    code_signals = [
        '.strip(', '.group(', '.replace(', '.split(',   # method calls
        'llm_query(', 'print(', 're.search(',             # function calls
        'f"', "f'",                                       # f-strings
        '\\n', 'elif ', 'else:', 'import ',               # control flow
        ' = ', '==', '!=',                                # assignments/comparisons
        '].', ').', '}{',                                 # chained operations
    ]
    return any(sig in text for sig in code_signals)


def _resolve_variable(name: str, repl_env) -> Optional[str]:
    """Try to resolve a variable name from the REPL environment."""
    name = name.strip().strip('"').strip("'").strip()
    # Check locals first, then globals
    if name in repl_env.locals:
        return str(repl_env.locals[name])
    if name in repl_env.globals:
        val = repl_env.globals[name]
        # Don't return functions/modules/builtins
        if isinstance(val, (str, int, float, bool)):
            return str(val)
    return None


def check_for_final_answer(response: str, repl_env, logger) -> Optional[str]:
    result = find_final_answer(response)
    if result is None:
        return None
    
    answer_type, content = result
    
    if answer_type == 'FINAL_VAR':
        # Resolve the variable from REPL
        resolved = _resolve_variable(content, repl_env)
        if resolved is not None:
            return resolved
        else:
            error_msg = f"Variable '{content}' not found in REPL environment"
            logger.log_tool_execution("FINAL_VAR", error_msg)
            return None
    
    elif answer_type == 'FINAL':
        # Check if content looks like code or a bare variable name
        if _looks_like_code(content):
            # The model wrote something like FINAL(answer.strip()) — try extracting the var name
            var_match = re.match(r'^(\w+)', content)
            if var_match:
                resolved = _resolve_variable(var_match.group(1), repl_env)
                if resolved is not None:
                    return resolved
            # Can't resolve — return None so the loop continues
            return None
        
        # Check if content is a bare Python identifier (like "final_answer" or "answer")
        if content.isidentifier() and len(content) > 2:
            resolved = _resolve_variable(content, repl_env)
            if resolved is not None:
                return resolved
            # Variable not found — it might genuinely be a one-word answer like "Yes" or "No"
            # Only return it if it looks like a plausible answer (short, capitalized)
            if len(content) <= 15:
                return content
            return None
        
        return content
    
    return None

def add_execution_result_to_messages(messages: List[Dict[str, str]], 
                                   code: str, 
                                   result: str,
                                   max_character_length: int = 100000,
                                   ) -> List[Dict[str, str]]:

    if len(result) > max_character_length:
        result = result[:max_character_length] + "..."
    
    execution_message = {
        "role": "user",
        "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result}"
    }
    messages.append(execution_message)
    return messages

def format_execution_result(
    stdout: str,
    stderr: str,
    locals_dict: Dict[str, Any],
    truncate_length: int = 100
) -> str:

    result_parts = []
    
    if stdout:
        result_parts.append(f"\n{stdout}")
    
    if stderr:
        result_parts.append(f"\n{stderr}")
    
    important_vars = {}
    for key, value in locals_dict.items():
        if not key.startswith('_') and not key in ['__builtins__', '__name__', '__doc__']:
            try:
                if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                    if isinstance(value, str) and len(value) > truncate_length:
                        important_vars[key] = f"'{value[:truncate_length]}...'"
                    else:
                        important_vars[key] = repr(value)
            except:
                important_vars[key] = f"<{type(value).__name__}>"
    
    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")
    
    return "\n\n".join(result_parts) if result_parts else "No output"


def execute_code(repl_env, code: str, repl_env_logger, logger) -> str:
    try:
        result = repl_env.code_execution(code)
        
        formatted_result = format_execution_result(
            result.stdout, result.stderr, result.locals
        )
        repl_env_logger.log_execution(code, result.stdout, result.stderr, result.execution_time)
        repl_env_logger.display_last()

        logger.log_tool_execution("CODE_EXECUTION", formatted_result)
        
        return formatted_result
        
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        return error_msg

def process_code_execution(
    response: str,
    messages: List[Dict[str, str]],
    repl_env,
    repl_env_logger,
    logger,
) -> List[Dict[str, str]]:
    code_blocks = find_code_blocks(response)
    
    if code_blocks:
        # Execute each code block
        for code in code_blocks:
            execution_result = execute_code(repl_env, code, repl_env_logger, logger)
            
            # Add execution result to conversation
            messages = add_execution_result_to_messages(
                messages, code, execution_result, 
            )
    
    return messages

def convert_context_for_repl(context):
    if isinstance(context, dict):
        context_data = context
        context_str = None
    elif isinstance(context, str):
        context_data = None
        context_str = context
    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            if "content" in context[0]:
                context_data = [msg.get("content", "") for msg in context]
            else:
                context_data = context
            context_str = None
        else:
            context_data = context
            context_str = None
    else:
        context_data = context
        context_str = None
    
    return context_data, context_str