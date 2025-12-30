# CodeQL labeling (folder-based run)

We ran CodeQL on **a folder of extracted Java Functions** (each saved as a `.java` file),
rather than running CodeQL on an entire upstream repository build.

## Tool versions
- CodeQL CLI: `2.23.5`
- Query pack: `codeql/java-queries@1.9.0`

## Folder-based analysis note
Because the source root is a folder of `.java` files, the database is created with:
- `--language=java`
- `--source-root <FUNCTION_FOLDER>`

## Query suites used
We used the standard Java suites from `codeql/java-queries`:
- `codeql-suites/java-security-and-quality.qls`
- `codeql-suites/java-security.qls` 

Exact interpretation of alerts →  a function is marked *vulnerable (1)* if it overlaps with at least one CodeQL alert location; otherwise *non-vulnerable (0)*
