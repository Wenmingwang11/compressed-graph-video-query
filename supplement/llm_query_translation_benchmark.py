from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


VALID_DISTANCES = {"any", "near", "medium", "far"}
VALID_DIRECTIONS = {"any", "left_of", "right_of", "above", "below"}


def resolve_codex_executable() -> str:
    executable_name = "codex.cmd" if os.name == "nt" else "codex"
    executable = shutil.which(executable_name)
    if executable is None:
        raise FileNotFoundError(f"could not locate {executable_name} on PATH")
    return executable


def safe_model_slug(model: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in model)


def chunk_cases(cases: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [cases[index : index + batch_size] for index in range(0, len(cases), batch_size)]


def _canonical_attributes(attributes: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(item["name"]), str(item["value"])) for item in attributes))


def _canonical_objects(objects: Iterable[Mapping[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                str(item["role"]),
                str(item["category"]),
                _canonical_attributes(item.get("attributes", [])),
            )
            for item in objects
        )
    )


def _canonical_relations(relations: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            (
                str(item["source"]),
                str(item["target"]),
                str(item["distance"]),
                str(item["direction"]),
            )
            for item in relations
        )
    )


def validate_prediction(prediction: Mapping[str, Any]) -> None:
    required = {"case_id", "objects", "relations", "DF", "k"}
    if required.difference(prediction):
        raise ValueError(f"missing fields: {sorted(required.difference(prediction))}")
    if not prediction["objects"]:
        raise ValueError("objects must not be empty")
    roles = [str(item["role"]) for item in prediction["objects"]]
    if len(roles) != len(set(roles)):
        raise ValueError("object roles must be unique")
    for relation in prediction["relations"]:
        if relation["source"] not in roles or relation["target"] not in roles:
            raise ValueError("relation references an unknown role")
        if relation["source"] == relation["target"]:
            raise ValueError("relation must connect different roles")
        if relation["distance"] not in VALID_DISTANCES:
            raise ValueError("unsupported distance label")
        if relation["direction"] not in VALID_DIRECTIONS:
            raise ValueError("unsupported direction label")
    if int(prediction["DF"]) <= 0 or int(prediction["k"]) <= 0:
        raise ValueError("DF and k must be positive")


def score_prediction(gold: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    score: dict[str, Any] = {
        "case_id": str(gold["case_id"]),
        "object_count": len(gold["objects"]),
        "format_valid": False,
        "objects_correct": False,
        "attributes_correct": False,
        "relations_correct": False,
        "df_correct": False,
        "k_correct": False,
        "exact_match": False,
    }
    try:
        validate_prediction(prediction)
        score["format_valid"] = True
    except (KeyError, TypeError, ValueError):
        return score

    gold_objects = _canonical_objects(gold["objects"])
    predicted_objects = _canonical_objects(prediction["objects"])
    score["objects_correct"] = tuple(item[:2] for item in gold_objects) == tuple(
        item[:2] for item in predicted_objects
    )
    score["attributes_correct"] = tuple((item[0], item[2]) for item in gold_objects) == tuple(
        (item[0], item[2]) for item in predicted_objects
    )
    score["relations_correct"] = _canonical_relations(gold["relations"]) == _canonical_relations(
        prediction["relations"]
    )
    score["df_correct"] = int(gold["DF"]) == int(prediction["DF"])
    score["k_correct"] = int(gold["k"]) == int(prediction["k"])
    score["exact_match"] = all(
        score[field]
        for field in (
            "format_valid",
            "objects_correct",
            "attributes_correct",
            "relations_correct",
            "df_correct",
            "k_correct",
        )
    )
    return score


def _percentage(rows: list[Mapping[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return round(100.0 * sum(bool(row[field]) for row in rows) / len(rows), 2)


def summarize_scores(scores: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for score in scores:
        grouped[str(score.get("model", "unknown"))].append(score)

    summaries: list[dict[str, Any]] = []
    for model, rows in sorted(grouped.items()):
        pairs = [row for row in rows if int(row["object_count"]) == 2]
        triples = [row for row in rows if int(row["object_count"]) == 3]
        wall_times = [float(row["batch_wall_seconds"]) for row in rows if "batch_wall_seconds" in row]
        summaries.append(
            {
                "model": model,
                "n_cases": len(rows),
                "format_valid_pct": _percentage(rows, "format_valid"),
                "exact_match_pct": _percentage(rows, "exact_match"),
                "m2_exact_match_pct": _percentage(pairs, "exact_match"),
                "m3_exact_match_pct": _percentage(triples, "exact_match"),
                "objects_pct": _percentage(rows, "objects_correct"),
                "attributes_pct": _percentage(rows, "attributes_correct"),
                "relations_pct": _percentage(rows, "relations_correct"),
                "df_pct": _percentage(rows, "df_correct"),
                "k_pct": _percentage(rows, "k_correct"),
                "median_batch_wall_seconds": round(statistics.median(wall_times), 3)
                if wall_times
                else "",
            }
        )
    return summaries


def _distance_bins(label: str, distance_parts: int) -> list[int]:
    if label == "any":
        return [0, distance_parts - 1]
    near_end = max(0, distance_parts // 4)
    medium_end = max(near_end + 1, (5 * distance_parts) // 8)
    if label == "near":
        return [0, near_end]
    if label == "medium":
        return [near_end + 1, min(distance_parts - 1, medium_end)]
    if label == "far":
        return [min(distance_parts - 1, medium_end + 1), distance_parts - 1]
    raise ValueError(f"unsupported distance label: {label}")


def _theta_bins(label: str, theta_parts: int) -> list[int]:
    quarter = max(1, theta_parts // 4)
    if label == "any":
        return list(range(-theta_parts, theta_parts + 1))
    if label == "left_of":
        return list(range(-quarter, quarter + 1))
    if label == "right_of":
        return list(range(-theta_parts, -theta_parts + quarter + 1)) + list(
            range(theta_parts - quarter, theta_parts + 1)
        )
    if label == "above":
        return list(range(quarter + 1, theta_parts - quarter + 1))
    if label == "below":
        return list(range(-theta_parts + quarter, -quarter))
    raise ValueError(f"unsupported direction label: {label}")


def compile_semantic_query(
    query: Mapping[str, Any],
    category_ids: Mapping[str, int],
    theta_parts: int = 10,
    distance_parts: int = 8,
) -> dict[str, Any]:
    validate_prediction(query)
    ordered_objects = sorted(query["objects"], key=lambda item: int(str(item["role"])[1:]))
    role_positions = {str(item["role"]): index for index, item in enumerate(ordered_objects)}
    role_types = [int(category_ids[str(item["category"])]) for item in ordered_objects]
    role_attributes = [
        {str(attribute["name"]): str(attribute["value"]) for attribute in item.get("attributes", [])}
        for item in ordered_objects
    ]
    relations = []
    for relation in query["relations"]:
        relations.append(
            {
                "source_role": role_positions[str(relation["source"])],
                "target_role": role_positions[str(relation["target"])],
                "distance_bins": _distance_bins(str(relation["distance"]), distance_parts),
                "theta_bins": _theta_bins(str(relation["direction"]), theta_parts),
            }
        )
    return {
        "query_id": str(query["case_id"]),
        "role_types": role_types,
        "role_attributes": role_attributes,
        "relations": relations,
        "min_consecutive_frames": int(query["DF"]),
        "topk": int(query["k"]),
        "theta_parts": int(theta_parts),
        "distance_parts": int(distance_parts),
    }


def output_schema() -> dict[str, Any]:
    attribute = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
        "required": ["name", "value"],
        "additionalProperties": False,
    }
    object_item = {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "category": {"type": "string"},
            "attributes": {"type": "array", "items": attribute},
        },
        "required": ["role", "category", "attributes"],
        "additionalProperties": False,
    }
    relation = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "distance": {"type": "string", "enum": sorted(VALID_DISTANCES)},
            "direction": {"type": "string", "enum": sorted(VALID_DIRECTIONS)},
        },
        "required": ["source", "target", "distance", "direction"],
        "additionalProperties": False,
    }
    prediction = {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "objects": {"type": "array", "items": object_item},
            "relations": {"type": "array", "items": relation},
            "DF": {"type": "integer"},
            "k": {"type": "integer"},
        },
        "required": ["case_id", "objects", "relations", "DF", "k"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"predictions": {"type": "array", "items": prediction}},
        "required": ["predictions"],
        "additionalProperties": False,
    }


def build_prompt(cases: list[Mapping[str, Any]]) -> str:
    inputs = [{"case_id": item["case_id"], "text": item["text"]} for item in cases]
    return (
        "Convert every natural-language video query below into the semantic query JSON required by "
        "the supplied output schema. Do not use tools and do not add cases. Assign roles r1, r2, r3 "
        "in order of first mention. Preserve object categories in lowercase singular form. Extract explicit "
        "attributes as name/value pairs. Allowed distance labels are any, near, medium, far. Allowed direction "
        "labels describe the source object's position relative to the target: any, left_of, right_of, above, "
        "below. If a query states only direction, use distance=any; if it states only distance, use direction=any. "
        "DF is the minimum number of consecutive frames and k is the requested result count. Return exactly one "
        "prediction per input case.\nINPUTS:\n"
        + json.dumps(inputs, ensure_ascii=True, separators=(",", ":"))
    )


def build_ollama_payload(model: str, cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(cases)}],
        "stream": False,
        "think": False,
        "format": output_schema(),
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_ctx": 16384,
            "num_predict": 2048,
        },
        "keep_alive": "10m",
    }


def run_codex_model(
    model: str,
    cases: list[Mapping[str, Any]],
    output_dir: Path,
    run_number: int,
    timeout_seconds: int,
) -> tuple[dict[str, Any], float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qst_llm_") as temp_dir:
        temp_root = Path(temp_dir)
        schema_path = temp_root / "schema.json"
        response_path = temp_root / "response.json"
        schema_path.write_text(json.dumps(output_schema(), ensure_ascii=True), encoding="utf-8")
        command = [
            resolve_codex_executable(),
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--ignore-rules",
            "-s",
            "read-only",
            "-m",
            model,
            "-c",
            'model_provider="openai"',
            "-c",
            'model_reasoning_effort="low"',
            "-c",
            'model_verbosity="low"',
            "--output-schema",
            str(schema_path),
            "-o",
            str(response_path),
            "--color",
            "never",
            "-",
        ]
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            input=build_prompt(cases),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        wall_seconds = time.perf_counter() - start
        model_slug = safe_model_slug(model)
        log_path = output_dir / f"{model_slug}_run{run_number}_codex.log"
        log_path.write_text(completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"{model} run {run_number} failed; see {log_path}")
        payload = json.loads(response_path.read_text(encoding="utf-8"))

    raw_path = output_dir / f"{model_slug}_run{run_number}_predictions.json"
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload, wall_seconds


def run_ollama_model(
    model: str,
    cases: list[Mapping[str, Any]],
    output_dir: Path,
    run_number: int,
    timeout_seconds: int,
    base_url: str,
    batch_size: int = 1,
) -> tuple[dict[str, Any], dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    envelopes: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    wall_seconds = 0.0
    model_slug = safe_model_slug("ollama_" + model)
    response_path = output_dir / f"{model_slug}_run{run_number}_ollama_response.json"
    prediction_path = output_dir / f"{model_slug}_run{run_number}_predictions.json"
    for case_batch in chunk_cases(cases, batch_size):
        case_ids = [str(case.get("case_id", "")) for case in case_batch]
        request = urllib.request.Request(
            base_url.rstrip("/") + "/api/chat",
            data=json.dumps(build_ollama_payload(model, case_batch), ensure_ascii=True).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            envelope["case_ids"] = case_ids
        except TimeoutError as error:
            envelope = {
                "case_ids": case_ids,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        wall_seconds += time.perf_counter() - start
        envelopes.append(envelope)
        try:
            batch_payload = json.loads(envelope["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError):
            batch_payload = {"predictions": []}
        predictions.extend(batch_payload.get("predictions", []))

        response_path.write_text(json.dumps(envelopes, indent=2, ensure_ascii=False), encoding="utf-8")
        prediction_path.write_text(
            json.dumps({"predictions": predictions}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    payload = {"predictions": predictions}
    total_model_seconds = sum(float(item.get("total_duration", 0)) for item in envelopes) / 1e9
    metrics = {
        "batch_wall_seconds": round(wall_seconds, 6),
        "model_total_seconds": round(total_model_seconds, 6),
        "mean_model_seconds_per_query": round(total_model_seconds / max(1, len(cases)), 6),
        "model_load_seconds": round(
            sum(float(item.get("load_duration", 0)) for item in envelopes) / 1e9,
            6,
        ),
        "prompt_eval_seconds": round(
            sum(float(item.get("prompt_eval_duration", 0)) for item in envelopes) / 1e9,
            6,
        ),
        "generation_seconds": round(
            sum(float(item.get("eval_duration", 0)) for item in envelopes) / 1e9,
            6,
        ),
        "prompt_tokens": sum(int(item.get("prompt_eval_count", 0)) for item in envelopes),
        "output_tokens": sum(int(item.get("eval_count", 0)) for item in envelopes),
        "failed_batches": sum(1 for item in envelopes if item.get("error_type")),
    }
    return payload, metrics


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(path: Path, summaries: list[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{自然语言查询转换结果（\%）}",
        r"\label{tab:nl-query-translation}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"模型 & 格式有效 & 完全正确 & $m=2$ & $m=3$ & 空间关系 & $DF/k$ \\ \hline",
    ]
    for row in summaries:
        lines.append(
            f"{row['model']} & {row['format_valid_pct']:.2f} & {row['exact_match_pct']:.2f} & "
            f"{row['m2_exact_match_pct']:.2f} & {row['m3_exact_match_pct']:.2f} & "
            f"{row['relations_pct']:.2f} & {min(row['df_pct'], row['k_pct']):.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLM translation into semantic QST queries.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("llm_query_translation_cases.json"),
    )
    parser.add_argument("--models", nargs="+", default=["gpt-5.5", "gpt-5.4-mini"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("out") / "llm_query_translation",
    )
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-batch-size", type=int, default=1)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    gold_by_id = {str(item["case_id"]): item["gold"] for item in cases}
    all_scores: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        for run_number in range(1, args.runs + 1):
            if model.startswith("ollama/"):
                payload, timing = run_ollama_model(
                    model.split("/", 1)[1],
                    cases,
                    args.output_dir,
                    run_number,
                    args.timeout_seconds,
                    args.ollama_url,
                    args.ollama_batch_size,
                )
            else:
                payload, wall_seconds = run_codex_model(
                    model,
                    cases,
                    args.output_dir,
                    run_number,
                    args.timeout_seconds,
                )
                timing = {
                    "batch_wall_seconds": round(wall_seconds, 6),
                    "model_total_seconds": "",
                    "mean_model_seconds_per_query": "",
                    "model_load_seconds": "",
                    "prompt_eval_seconds": "",
                    "generation_seconds": "",
                    "prompt_tokens": "",
                    "output_tokens": "",
                }
            predictions = {str(item.get("case_id")): item for item in payload.get("predictions", [])}
            for case_id, gold in gold_by_id.items():
                prediction = predictions.get(case_id, {})
                score = score_prediction(gold, prediction)
                score.update(
                    {
                        "model": model,
                        "run": run_number,
                        **timing,
                    }
                )
                all_scores.append(score)

    summaries = summarize_scores(all_scores)
    _write_csv(args.output_dir / "llm_query_translation_scores.csv", all_scores)
    _write_csv(args.output_dir / "llm_query_translation_summary.csv", summaries)
    _write_latex(args.output_dir / "llm_query_translation_table.tex", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
