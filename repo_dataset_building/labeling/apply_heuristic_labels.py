import re
import pandas as pd
from heuristic_patterns import CWE_PATTERNS

INPUT_CSV = "input.csv"
OUTPUT_CSV = "output_with_heuristic_cwes.csv"
CODE_COL = "old_func"

def match_cwes(code: str):
    out = []
    if not isinstance(code, str) or not code.strip():
        return out
    for cwe, patterns in CWE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, code, re.IGNORECASE):
                out.append(cwe)
                break
    return out

def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df["cwe_ids"] = df[CODE_COL].fillna("").apply(lambda s: ",".join(match_cwes(s)) or "None")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
