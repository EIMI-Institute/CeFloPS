import os
import ast

def extract_imports(filepath):
    """Extracts import statements from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        return imports
    except SyntaxError as e:
        print(f"SyntaxError in file: {filepath}")
        print(f"Error message: {e}")
        return []

def analyze_project(project_root):
    """Analyzes a project's Python files and extracts import information."""
    import_data = {}
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                imports = extract_imports(filepath)
                if imports:
                    import_data[filepath] = imports

    return import_data

def print_import_structure(import_data):
    """Prints the import structure in a hierarchical format."""
    for filepath, imports in import_data.items():
        print(f"File: {filepath}")
        for import_item in imports:
            print(f"  - {import_item}")
        print("-" * 30)

# --- Main Execution ---
if __name__ == "__main__":
    project_root = input("Enter the root directory of your project: ")  # Get project root from user
    import_data = analyze_project(project_root)
    print_import_structure(import_data)
