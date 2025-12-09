import json
import sys

def clear_notebook(notebook_path):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Clear outputs from ALL cells
    for cell in notebook['cells']:
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
        
        # Also clear any metadata that might contain outputs
        if 'metadata' in cell:
            # Remove execution metadata
            if 'execution' in cell['metadata']:
                del cell['metadata']['execution']
    
    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Cleared outputs from {notebook_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        clear_notebook(sys.argv[1])
    else:
        print("Usage: python clear_outputs.py <notebook_file.ipynb>")