# Joern-based CPG (GraphSON) Export

We export Code Property Graphs (CPGs) for each sample using **Joern v4.0.423** and serialize them in **GraphSON** format for downstream processing.

## Tooling
- Joern: **4.0.423** 
- Commands used: `joern-parse`, `joern-export`
- References: Joern overview + export docs. (See: docs.joern.io) :contentReference[oaicite:1]{index=1}

## Script
- `export_graphson.sh` iterates over `Sample_0000 ... Sample_XXXX`
- For each sample `Sample_XXXX`:
  - Copies the source file into `Sample_XXXX/`
  - Produces `Sample_XXXX/cpg.bin` via `joern-parse`
  - Exports GraphSON to `Sample_XXXX/cpg_graphson/` via:
    - `joern-export cpg.bin --repr all --format graphson --out cpg_graphson`

## Output layout (per sample)
Sample_0001/
  ├── Sample_0001.java
  ├── cpg.bin
  └── cpg_graphson/
      └── ... GraphSON files ...

- GraphSON is used as a portable JSON graph interchange format for CPG data.