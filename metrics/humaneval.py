import os
import sys
import json
import ast
import traceback
import glob
import re
from typing import Dict, List, Optional, Set, Tuple
import evaluate as hf_evaluate
from transformers import AutoTokenizer
import argparse


os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def refine_text(text: str) -> str:
    text = text.replace("\t", "    ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > 100:
        lines = lines[:100]
    max_valid_lines = 0
    max_valid_snippet = ""
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            current_snippet = "\n".join(lines[i : j + 1])
            if syntax_check(current_snippet):
                valid_line_count = sum(1 for line in lines[i : j + 1] if line.strip())
                if valid_line_count > max_valid_lines:
                    max_valid_lines = valid_line_count
                    max_valid_snippet = current_snippet
    return max_valid_snippet


def get_deps(nodes: List[Tuple[str, ast.AST]]) -> Dict[str, Set[str]]:
    name2deps = {}
    for name, node in nodes:
        deps = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for child in ast.iter_child_nodes(current):
                if isinstance(child, ast.Name):
                    deps.add(child.id)
                elif isinstance(child, ast.Attribute):
                    pass
                else:
                    stack.append(child)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    visited = set()
    to_visit = [entrypoint]
    while to_visit:
        current = to_visit.pop(0)
        if current not in visited:
            visited.add(current)
            to_visit.extend(call_graph.get(current, set()) - visited)
    return visited


def get_definition_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    elif isinstance(node, ast.Assign):
        targets = node.targets
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
    return None


def has_return_statement(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(node))


def sanitize(text: str, entrypoint: Optional[str] = None) -> str:
    text = refine_text(text)
    try:
        code = extract_longest_valid_code(text)
        if not code:
            return ""
        tree = ast.parse(code)
    except (SyntaxError, MemoryError):
        return ""
    definitions = {}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            name = node.name
            definitions[name] = ("class", node)
        elif isinstance(node, ast.FunctionDef):
            name = node.name
            if has_return_statement(node):
                definitions[name] = ("function", node)
        elif isinstance(node, ast.Assign):
            name = get_definition_name(node)
            if name:
                definitions[name] = ("variable", node)
    if entrypoint:
        name2deps = get_deps([(name, node) for name, (_, node) in definitions.items()])
        reachable = get_function_dependency(entrypoint, name2deps)
    else:
        reachable = set(definitions.keys())
    sanitized_output = []
    for node in imports:
        sanitized_output.append(ast.unparse(node))
    for name, (_, node) in definitions.items():
        if name in reachable:
            sanitized_output.append(ast.unparse(node))
    return "\n".join(sanitized_output)


def count_total_tokens(text, tokenizer):
    if not text or not tokenizer:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_special_tokens(text: str) -> int:
    special_tokens_pattern = r"\[DONE\]|<\|eot_id\|>|<\|endoftext\|>"
    return len(re.findall(special_tokens_pattern, text))


def evaluate_humaneval_results(directory, tokenizer_path):
    print("\n" + "="*50 + f"\nProcessing HumanEval directory: {directory}\n" + "="*50)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Failed to load Tokenizer, please check the path: {e}")
        return
    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No .jsonl files found in directory '{directory}'.")
        return
    all_predictions, all_references = [], []
    agg_stats = {
        "processed": 0,
        "total_raw_tokens": 0,
        "total_special_tokens": 0,
    }

    print(f"Found {len(jsonl_files)} files to process...")
    for file_path in jsonl_files:
        print(f"  -> Processing file: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:                    
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    raw_generation = item['resps'][0][0]
                    prompt = item["doc"]["prompt"]
                    entry_point = item["doc"]["entry_point"]
                    reference = item["target"]
                    agg_stats["total_raw_tokens"] += count_total_tokens(raw_generation, tokenizer)
                    agg_stats["total_special_tokens"] += count_special_tokens(raw_generation)
                    code_to_sanitize = raw_generation.split("```python\n", 1)[-1].split("```")[0]
                    full_text = prompt + "\n" + code_to_sanitize
                    sanitized_code = sanitize(full_text, entry_point)
                    all_predictions.append([sanitized_code])
                    all_references.append(reference)
                    agg_stats["processed"] += 1
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"    Error processing file '{os.path.basename(file_path)}': {e}")
            continue
    total_processed = agg_stats["processed"]
    print(f"\nLoading the code_eval evaluator and starting evaluation...")
    code_eval = hf_evaluate.load("code_eval")
    pass_at_k_results, _ = code_eval.compute(
        references=all_references,
        predictions=all_predictions,
        k=[1],
        num_workers=max(1, os.cpu_count() // 2)
    )
    pass_1_score = pass_at_k_results.get("pass@1", 0.0)
    correct_answers = round(pass_1_score * total_processed)
    if total_processed > 0:
        accuracy = pass_1_score * 100
        avg_len = agg_stats["total_raw_tokens"] / total_processed
        effective_tokens = agg_stats["total_raw_tokens"] - agg_stats["total_special_tokens"]
        avg_effective_len = effective_tokens / total_processed if total_processed > 0 else 0
        eot_prop = (agg_stats["total_special_tokens"] / agg_stats["total_raw_tokens"] * 100) if agg_stats["total_raw_tokens"] > 0 else 0
    else:
        print("No valid data processed. Cannot calculate results.")
        return
    print("\n" + "-" * 80)
    print(f"Results for '{os.path.basename(directory)}'")
    print("-" * 80)
    print(f"  - Accuracy:               {accuracy:.2f}%")
    print(f"  - Avg. Effective Tokens: {avg_effective_len:.2f}")
    print(f"  - Avg. Total Tokens:   {avg_len:.2f}")
    print(f"  - Avg. Effective Tokens Ratio: {(100-eot_prop):.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", 
        type=str, 
        required=True)
    parser.add_argument(
        "-r", "--res_path", 
        type=str, 
        required=True)
    args = parser.parse_args()
    
    evaluate_humaneval_results(directory=args.res_path, tokenizer_path=args.model_path)
    