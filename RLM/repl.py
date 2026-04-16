import sys
import io
import threading
import json
import tempfile
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from RLM.rlm import RLM
from RLM.utils.llm import DEFAULT_MODEL

# Simple sub LM for REPL environment. Note: This could also be just the RLM itself!
class Sub_RLM(RLM):
    """Recursive LLM client for REPL environment with fixed configuration."""
    
    def __init__(self, model: str = None):
        model = model or DEFAULT_MODEL
        # support either OpenAI or Gemini API key environment variables
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GENAI_API_KEY")
        if not self.api_key and not model.lower().startswith("ollama/"):
            raise ValueError("API key required: set OPENAI_API_KEY or GENAI_API_KEY")
        
        self.model = model

        # Initialize a generic LLM client (provider determined by model name)
        from RLM.utils.llm import LLMClient
        self.client = LLMClient(api_key=self.api_key, model=model)
        
    
    def completion(self, prompt) -> str:
        """
        Simple LM query for sub-LM call.
        """
        try:
            # Handle both string and dictionary/list inputs
            response = self.client.completion(
                messages=prompt,
                timeout=300
            )
            
            return response
                
        except Exception as e:
            error_msg = f"Error making LLM query: {str(e)}"
            return error_msg
    
    def cost_summary(self) -> dict[str, float]:
        raise NotImplementedError("Cost tracking is not implemented for the Sub-RLM.")
    
    def reset(self):
        raise NotImplementedError("Reset is not implemented for the Sub-RLM.")


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float
    
    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, locals={self.locals}, execution_time={self.execution_time})"

class REPLEnv:
    def __init__(
        self,
        recursive_model: str = None,
        context_json: Optional[dict | list] = None,
        context_str: Optional[str] = None,
        setup_code: str = None,
        plugins: Optional[dict] = None,
    ):
        # Store the original working directory
        self.original_cwd = os.getcwd()
        
        # Create temporary directory (but don't change global working directory)
        self.temp_dir = tempfile.mkdtemp(prefix="repl_env_")


        # Initialize minimal RLM / LM client. Change this to support more depths.
        self.sub_rlm: RLM = Sub_RLM(model=recursive_model)
        
        # Create safe globals with only string-safe built-ins
        self.globals = {
            '__builtins__': {
                # Safe built-ins for string manipulation
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
                'type': type, 'isinstance': isinstance, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin, 'oct': oct,
                'repr': repr, 'ascii': ascii, 'format': format,
                '__import__': __import__,  # Allow imports
                'open': open,  # Allow file access
                
                # Add commonly used built-ins that were missing
                'any': any, 'all': all, 'hasattr': hasattr, 'getattr': getattr,
                'setattr': setattr, 'delattr': delattr, 'dir': dir, 'vars': vars,
                'range': range,  # Add range function
                'reversed': reversed,  # Add reversed function
                'slice': slice,  # Add slice function
                'iter': iter,  # Add iter function
                'next': next,  # Add next function
                'pow': pow,  # Add pow function
                'divmod': divmod,  # Add divmod function
                'complex': complex,  # Add complex function
                'bytes': bytes,  # Add bytes function
                'bytearray': bytearray,  # Add bytearray function
                'memoryview': memoryview,  # Add memoryview function
                'hash': hash,  # Add hash function
                'id': id,  # Add id function
                'callable': callable,  # Add callable function
                'issubclass': issubclass,  # Add issubclass function
                'super': super,  # Add super function
                'property': property,  # Add property function
                'staticmethod': staticmethod,  # Add staticmethod function
                'classmethod': classmethod,  # Add classmethod function
                'object': object,  # Add object class
                'BaseException': BaseException,  # Add BaseException class
                'ArithmeticError': ArithmeticError,  # Add ArithmeticError class
                'LookupError': LookupError,  # Add LookupError class
                'EnvironmentError': EnvironmentError,  # Add EnvironmentError class
                'AssertionError': AssertionError,  # Add AssertionError class
                'NotImplementedError': NotImplementedError,  # Add NotImplementedError class
                'UnicodeError': UnicodeError,  # Add UnicodeError class
                'Warning': Warning,  # Add Warning class
                'UserWarning': UserWarning,  # Add UserWarning class
                'DeprecationWarning': DeprecationWarning,  # Add DeprecationWarning class
                'PendingDeprecationWarning': PendingDeprecationWarning,  # Add PendingDeprecationWarning class
                'SyntaxWarning': SyntaxWarning,  # Add SyntaxWarning class
                'RuntimeWarning': RuntimeWarning,  # Add RuntimeWarning class
                'FutureWarning': FutureWarning,  # Add FutureWarning class
                'ImportWarning': ImportWarning,  # Add ImportWarning class
                'UnicodeWarning': UnicodeWarning,  # Add UnicodeWarning class
                'BytesWarning': BytesWarning,  # Add BytesWarning class
                'ResourceWarning': ResourceWarning,  # Add ResourceWarning class
                
                # Add exception classes
                'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
                'KeyError': KeyError, 'IndexError': IndexError, 'AttributeError': AttributeError,
                'FileNotFoundError': FileNotFoundError, 'OSError': OSError, 'IOError': IOError,
                'RuntimeError': RuntimeError, 'NameError': NameError, 'ImportError': ImportError,
                'StopIteration': StopIteration, 'GeneratorExit': GeneratorExit,
                'SystemExit': SystemExit, 'KeyboardInterrupt': KeyboardInterrupt,

                # Disallow the following built-ins
                'input': None,  # Block input
                'eval': None,  # Block eval
                'exec': None,  # Block exec
                'compile': None,  # Block compile
                'globals': None,  # Block globals access
                'locals': None,  # Block locals access
            }
        }
        self.locals = {}
        self._lock = threading.Lock()
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()

        self.load_context(context_json, context_str)

        # Pre-inject commonly used stdlib modules so model code doesn't crash
        import re as _re
        import json as _json
        import math as _math
        import collections as _collections
        self.globals['re'] = _re
        self.globals['json'] = _json
        self.globals['math'] = _math
        self.globals['collections'] = _collections
        
        def llm_query(*args, **kwargs) -> str:
            """
            Query the LLM with the given prompt. Returns a concise text answer (string).
            
            IMPORTANT: This function returns a raw string. It does NOT have an '.answer' attribute.
            Usage: fact = llm_query("What is X?")
            """
            # Heuristic: Add a "concise" system constraint to prevent conversational leakage
            query_prompt = [
                {"role": "system", "content": "You are a factual assistant. Provide a direct, minimal answer to the user's query. If you cannot find the answer, say 'Answer not found'."},
                {"role": "user", "content": " ".join(str(arg) for arg in args)}
            ]
            return self.sub_rlm.completion(query_prompt)
        
        # Add (R)LM query function to globals
        self.globals['llm_query'] = llm_query

        def search_context(query: str, n_results: int = 3, snippet_length: int = 300, **kwargs) -> str:
            """
            Search for snippets in the context that match the query.
            Returns a formatted string of matched snippets.
            """
            ctx = self.globals.get('context', '')
            if not isinstance(ctx, str) or not ctx:
                return "Error: context is not available as a string for searching."
            
            # Simple keyword-based snippet extraction
            query_words = query.lower().split()
            sentences = ctx.split('. ')
            
            scored_sentences = []
            for sent in sentences:
                score = sum(1 for word in query_words if word in sent.lower())
                if score > 0:
                    scored_sentences.append((score, sent))
            
            # Sort by score descending
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            
            top_matches = [s[1] for s in scored_sentences[:n_results]]
            
            if not top_matches:
                return "No direct matches found in context."
            
            result = "Matched snippets from context:\n"
            for i, snippet in enumerate(top_matches):
                # Clean up and truncate snippet if needed (though sentences are usually short enough)
                clean_snippet = snippet.strip().replace('\n', ' ')
                result += f"{i+1}. [...]{clean_snippet}[...]\n"
            
            return result
            
        self.globals['search_context'] = search_context
        
        # Add FINAL_VAR function to globals
        def final_var(variable_name: str) -> str:
            """
            Return the value of a variable from the REPL environment as a final answer.
            This function is used by the model to return variables as final answers.
            """
            # Strip spaces, quotes, and newlines from variable name
            variable_name = variable_name.strip().strip('"').strip("'").strip('\n').strip('\r')
            try:
                # Check if variable exists in locals
                if variable_name in self.locals:
                    value = self.locals[variable_name]
                    return str(value)
                else:
                    return f"Error: Variable '{variable_name}' not found in REPL environment"
            except Exception as e:
                return f"Error retrieving variable '{variable_name}': {str(e)}"
        self.globals['FINAL_VAR'] = final_var
        
        # Inject a dummy FINAL function to catch when the model uses it incorrectly inside code
        def final_error(*args, **kwargs):
            raise RuntimeError("'FINAL()' should NOT be used inside Python code blocks! Output your final answer as raw text directly.")
        self.globals['FINAL'] = final_error

        # Inject any additional plugin functions (e.g. memory_retrieve, deep_reason)
        if plugins:
            for name, fn in plugins.items():
                self.globals[name] = fn
        
        # Finally, run any setup code if provided
        if setup_code:
            self.code_execution(setup_code)
    
    def load_context(self, context_json: Optional[dict | list] = None, context_str: Optional[str] = None):
        # Write context JSON to temporary directory using absolute (temp dir) path
        if context_json is not None:
            context_path = os.path.join(self.temp_dir, "context.json")
            with open(context_path, "w") as f:
                json.dump(context_json, f, indent=2)
            context_code = (
                f"import json\n"
                f"with open(r'{context_path}', 'r') as f:\n"
                f"    context = json.load(f)\n"
            )
            self.code_execution(context_code)
        
        if context_str is not None:
            context_path = os.path.join(self.temp_dir, "context.txt")
            with open(context_path, "w") as f:
                f.write(context_str)
            context_code = (
                f"import os\n"
                f"with open(r'{context_path}', 'r') as f:\n"
                f"    context = f.read()\n"
            )
            self.code_execution(context_code)
    
    def __del__(self):
        """Clean up temporary directory when object is destroyed"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass 
    
    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr"""
        with self._lock:
            # Store original streams
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Create new buffers for this execution
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            try:
                # Redirect streams
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                yield stdout_buffer, stderr_buffer
            finally:
                # Restore original streams
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    
    @contextmanager
    def _temp_working_directory(self):
        """Context manager to temporarily change working directory for REPL execution"""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)
    
    def code_execution(self, code) -> REPLResult:
        """
        Simple code execution "notebook-style" in a REPL environment.
        """
        start_time = time.time()
        with self._capture_output() as (stdout_buffer, stderr_buffer):
            with self._temp_working_directory():
                try:
                    # Split code into lines
                    lines = code.split('\n')
                    import_lines = []
                    other_lines = []
                    
                    for line in lines:
                        if line.startswith(('import ', 'from ')) and not line.startswith('#'):
                            import_lines.append(line)
                        else:
                            other_lines.append(line)
                    
                    # Execute imports first in globals to make them available
                    if import_lines:
                        import_code = '\n'.join(import_lines)
                        exec(import_code, self.globals, self.globals)
                    
                    # Execute the rest of the code. We also want to support expression printing.
                    if other_lines:
                        # Clean up lines for expression detection
                        non_comment_lines = []
                        for line in other_lines:
                            stripped = line.strip()
                            if stripped and not stripped.startswith('#'):
                                non_comment_lines.append(line)
                        
                        if non_comment_lines:
                            last_line = non_comment_lines[-1]
                            
                            # Check if the last line looks like an expression (not a statement)
                            is_expression = (
                                not last_line.strip().startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ', 'return ', 'yield ', 'break', 'continue', 'pass')) and
                                '=' not in last_line.split('#')[0] and  # Not an assignment
                                not last_line.strip().endswith(':') and  # Not a control structure
                                not last_line.strip().startswith('print(')  # Not an explicit print
                            )
                            
                            if is_expression:
                                # Execute everything except the last line
                                head_code = '\n'.join(other_lines[:-1])
                                if head_code.strip():
                                    exec(head_code, self.globals, self.locals)
                                
                                # Evaluate the last line
                                result = eval(last_line, self.globals, self.locals)
                                if result is not None:
                                    print(result)
                            else:
                                # Just execute the whole block
                                other_code = '\n'.join(other_lines)
                                exec(other_code, self.globals, self.locals)
                                
                except Exception as e:
                    error_msg = str(e)
                    
                    # INJECT HINTS BASED ON ERROR TYPE
                    hint = ""
                    if isinstance(e, AttributeError) and "attribute 'answer'" in error_msg:
                        hint = "\nHINT: llm_query() returns a raw STRING, not an object. Don't use '.answer'. Use 'result = llm_query(...)' directly."
                    elif isinstance(e, NameError) and "analysis" in error_msg:
                        hint = "\nHINT: You used 'analysis' before defining it. Did you mean to define it in the REPL first?"
                    elif isinstance(e, TypeError) and "unexpected keyword argument 'text'" in error_msg:
                        hint = "\nHINT: search_context() uses 'context' automatically. Don't pass 'text=context' to it."
                    elif isinstance(e, SyntaxError):
                        hint = "\nHINT: Check if you used an f-string (e.g., f\"{context}\"), it likely crashed or failed. Use \"...\" + context or llm_query(\"prompt\", context) instead."
                    
                    sys.stderr.write(f"{type(e).__name__}: {error_msg}{hint}\n")
        
        execution_time = time.time() - start_time
        return REPLResult(
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            locals=self.locals.copy(),
            execution_time=execution_time
        )
    
    def get_cost_summary(self):
        raise NotImplementedError("Cost tracking is not implemented for the REPL Environment.")