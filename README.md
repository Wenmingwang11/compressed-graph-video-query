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
├── baseline/            # Baseline integration
├── supplement/          # Experiment, evaluation, and plotting utilities
├── tests/               # Research tests
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

## Query Specification

Query definitions use `QuerySpec` and `RelationConstraint` in `supplement/corrected_query_model.py`.

| Field | Description |
| --- | --- |
| `role_types` | Required object categories for the query roles |
| `relations` | Distance-bin intervals and orientation bins between roles |
| `min_consecutive_frames` | Minimum consecutive matching frames |
| `topk` | Maximum results requested in Top-k mode |
| `theta_parts` | Orientation quantization parameter |
| `distance_parts` | Distance quantization parameter |

Spatial constraints use quantized bins. Map pixel thresholds using the same resolution and quantization settings as the index.

## Experiments

The `supplement/` directory organizes scripts for index statistics, query benchmarks, baseline comparisons, discretization studies, and visualization. Configure dataset paths and baseline environments for the local machine before running an experiment.

Evaluation should state query definitions, data scope, annotation source, result granularity, and matching rules. For clip-level precision and recall, predicted and reference intervals can be matched one-to-one under a specified temporal intersection-over-union threshold.

## Citation

If you use this work, please cite the associated manuscript submitted to EDBT. Final bibliographic information will be added when available.

## Contact

For questions and suggestions, please open an issue in this repository.
