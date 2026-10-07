# Compressed Graph Video Query

> Research repository accompanying our manuscript on structured video querying with incremental compressed graphs, submitted to **EDBT**.

## Overview

Compressed Graph Video Query studies efficient spatiotemporal retrieval over tracked objects in videos. Frames are represented as graphs whose vertices describe objects and whose edges encode spatial relationships. Structured queries combine object categories, distance and orientation constraints, and temporal requirements to retrieve video intervals.

## Environment Setup

```bash
git clone https://github.com/Wenmingwang11/compressed-graph-video-query.git
cd compressed-graph-video-query
python -m venv .venv
```

Activate the environment on your platform, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Project Structure

```text
compressed-graph-video-query/
├── vsimsearch/          # Graph, data, index, and query modules
├── tests/               # Index and query tests
├── prefix_tree.py       # Prefix tree module
├── Tree_Node.py         # Tree node module
├── FPgrowth.py          # Frequent pattern mining module
├── subsets.py           # Subset processing module
├── query.py             # Query entry point
├── batch_query.py       # Batch query entry point
├── paper_query*.py      # Paper query entry points
├── requirements.txt     # Python dependencies
└── README.md
```

## Data Format

The multi-class tracking reader expects comma-separated records with these columns, without a header:

```text
frame,id,left,top,width,height,conf,class,x,y,z
```

Each record describes one tracked object in a frame. Bounding boxes use pixel coordinates. Object identities connect observations across frames, and class identifiers specify object categories. The auxiliary `x`, `y`, and `z` fields are discarded by this reader.

Configure video resolution, class mapping, frame rate, and local data paths consistently for each dataset.

## Citation

If you use this work, please cite the associated manuscript submitted to EDBT. Final bibliographic information will be added when available.

## Contact

For questions and suggestions, please open an issue in this repository.
