import evaluate as hf_evaluate
import os
import re

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]

def build_predictions_instruct_delimiter(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    # # Regex: capture from "def" until the first "return" line
    # pattern = r"(def\s+\w+\([^)]*\):[\s\S]*?return[^\n]*)"

    # results = []
    # for r in resps:
    #     try:
    #         parts = r.split("```")
    #         if len(parts) < 2:
    #             results.append(r)  # fallback to original
    #             continue
    #         code_block = parts[1]
    #         if 'python\n' in code_block:
    #             results.append(code_block[7:])  # skip 'python\n'
    #         else:
    #             results.append(code_block)
    #     except Exception:
    #         results.append(r)  # fallback if anything fails

    # return results
    pattern = r"```python\n(.*?)```"

    results = [
        [
            re.search(pattern, r, re.DOTALL).group(1) if re.search(pattern, r, re.DOTALL) else r.split("```")[1] #r.split("```")[-2][7:]
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]

    return results