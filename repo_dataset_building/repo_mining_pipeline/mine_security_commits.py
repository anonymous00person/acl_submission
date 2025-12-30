import pandas as pd
from tqdm import tqdm

from repo_config import REPOS, KEYWORDS, BASE_DIR, COMMITS_CSV
from utils import clone_or_open, compile_keyword_regex, is_security_message

def main() -> None:
    rx = compile_keyword_regex(KEYWORDS)
    rows = []

    for slug in REPOS:
        repo = clone_or_open(slug, BASE_DIR)
        for c in tqdm(list(repo.iter_commits()), desc=f"scan {slug}"):
            msg = (c.message or "").strip()
            if not is_security_message(msg, rx):
                continue
            rows.append({
                "repo": slug,
                "repo_name": slug.replace("/", "_"),
                "sha": c.hexsha,
                "author": getattr(c.author, "name", ""),
                "date": c.committed_datetime.isoformat(),
                "message": msg,
            })

    df = pd.DataFrame(rows)
    df.to_csv(COMMITS_CSV, index=False)
    print(f"saved: {COMMITS_CSV} | rows={len(df)}")

if __name__ == "__main__":
    main()
