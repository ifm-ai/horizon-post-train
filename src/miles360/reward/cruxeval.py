import re
from .utils import check_correctness

def compute_score(model_output: str, ground_truth: str, extra_info: any = None) -> dict:
    model_output = str(model_output)
    # print(f">>> {model_output}")
    try:
        if "</think>" in model_output:
            # remove content until </think>
            model_output = re.split(r"</think>", model_output)[1]
        else:
            model_output = model_output
        # remove content between ```python and ```
        model_output = re.split(r"```python", model_output)[1]
        model_output = re.split(r"```", model_output)[0]
    except:
        model_output = model_output

    full_code = eval(ground_truth)["functional"] + "\n" + model_output
    score = 1.0 if check_correctness(full_code) else 0.0
    is_correct = score == 1.0
    return {"score": score, "acc": is_correct}