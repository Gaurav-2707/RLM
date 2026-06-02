import os
import json
import glob

def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    files = glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)
    
    if not files:
        print(f"No result files found in {results_dir}.")
        return

    print(f"=== Validating {len(files)} JSON traces ===")
    
    errors = 0
    warnings = 0

    for file in files:
        filename = os.path.basename(file)
        try:
            with open(file, "r") as f:
                data = json.load(f)
                
            meta = data.get("metadata", {})
            repl_history = data.get("repl_history", [])
            
            # Checks
            if "gold_answer" not in meta:
                print(f"[ERROR] {filename}: Missing gold_answer in metadata.")
                errors += 1
                
            if not repl_history:
                print(f"[WARNING] {filename}: Empty repl_history.")
                warnings += 1
            else:
                for step in repl_history:
                    if "snapshot_answer" not in step:
                        print(f"[ERROR] {filename}: Missing snapshot_answer at iteration {step.get('iteration')}")
                        errors += 1
                    if "confidence" not in step:
                        print(f"[WARNING] {filename}: Missing confidence at iteration {step.get('iteration')}")
                        warnings += 1

        except json.JSONDecodeError:
            print(f"[ERROR] {filename}: Invalid JSON format.")
            errors += 1
        except Exception as e:
            print(f"[ERROR] {filename}: Could not read file. {e}")
            errors += 1

    print("\n--- Validation Summary ---")
    if errors == 0 and warnings == 0:
        print("All traces passed validation successfully! Data is clean.")
    else:
        print(f"Validation finished with {errors} Errors and {warnings} Warnings.")
        print("Please review the logs above before running analysis scripts.")

if __name__ == "__main__":
    main()
