"""
==========================================================
Vision Analysis Evaluation
==========================================================
Evaluates the VisionAnalysisAgent (real YOLO detection) separately
from the text-RAG eval in run_eval.py -- detection quality needs
different metrics entirely: precision/recall/count-accuracy against
ground-truth annotations, not lexical F1 against a text answer.

Two modes:

1. Annotated mode (recommended) -- you provide ground-truth boxes
   in eval/vision_eval_set.json (COCO-style or simplified format
   below). Computes count accuracy, per-class precision/recall,
   and mean IoU on matched detections.

2. Unannotated smoke-test mode -- if no ground truth file exists,
   just runs detection on every image in data/ and reports latency,
   detection counts, and confidence distribution. Useful for a
   quick "is it working" check without building a labeled set.

Usage:
    python eval/eval_vision.py                       # smoke test
    python eval/eval_vision.py --annotated            # full metrics
    python eval/eval_vision.py --annotated --out eval/vision_results
==========================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.vision_analysis import VisionAnalysisAgent
from utils.logger import get_logger

logger = get_logger("eval_vision")

VISION_EVAL_SET_PATH = Path(__file__).with_name("vision_eval_set.json")

# ── Ground truth format (eval/vision_eval_set.json) ──────────────────────────
# [
#   {
#     "image": "sample1.jpg",
#     "ground_truth_boxes": [
#       {"class": "person", "x1": 34, "y1": 12, "x2": 210, "y2": 400},
#       {"class": "car",    "x1": 250, "y1": 100, "x2": 480, "y2": 300}
#     ]
#   },
#   ...
# ]
# Draw boxes with any annotation tool (LabelImg, CVAT, Roboflow's own
# annotation UI, or just eyeball pixel coordinates on a small image).


def iou(box_a: dict, box_b: dict) -> float:
    xa1, ya1 = max(box_a["x1"], box_b["x1"]), max(box_a["y1"], box_b["y1"])
    xa2, ya2 = min(box_a["x2"], box_b["x2"]), min(box_a["y2"], box_b["y2"])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(predicted: list[dict], ground_truth: list[dict], iou_threshold: float = 0.5) -> dict:
    """
    Greedy matching: each ground-truth box matched to the highest-IoU
    unmatched prediction of the same class, above iou_threshold.
    Returns TP/FP/FN counts and mean IoU on matches.
    """
    matched_gt = set()
    matched_pred = set()
    ious = []

    pairs = []
    for gi, gt in enumerate(ground_truth):
        for pi, pred in enumerate(predicted):
            if pred["class"] != gt["class"]:
                continue
            score = iou(pred, gt)
            if score >= iou_threshold:
                pairs.append((score, gi, pi))
    pairs.sort(reverse=True)

    for score, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        ious.append(score)

    tp = len(matched_gt)
    fn = len(ground_truth) - tp
    fp = len(predicted) - len(matched_pred)

    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
    }


def run_smoke_test(agent: VisionAnalysisAgent) -> list[dict]:
    images = sorted(
        f for f in Path(config.DATA_DIR).glob("*")
        if f.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    if not images:
        print(f"No image files found in {config.DATA_DIR}. Upload some images first, or")
        print("build eval/vision_eval_set.json for annotated evaluation.")
        sys.exit(1)

    print(f"Running detection smoke test on {len(images)} image(s)…")
    rows = []
    for img_path in images:
        result = agent.analyze_bytes(img_path.read_bytes(), source_label=img_path.name)
        rows.append({
            "image": img_path.name,
            "detection_count": len(result.get("detections", [])),
            "inference_seconds": result.get("inference_seconds", 0),
            "avg_confidence": (
                round(sum(d["confidence"] for d in result["detections"]) / len(result["detections"]), 3)
                if result.get("detections") else None
            ),
            "classes_detected": sorted(set(d["class"] for d in result.get("detections", []))),
        })
        print(f"  {img_path.name}: {rows[-1]['detection_count']} detection(s), "
              f"{rows[-1]['inference_seconds']}s")
    return rows


def run_annotated_eval(agent: VisionAnalysisAgent) -> dict:
    if not VISION_EVAL_SET_PATH.exists():
        print(f"ERROR: {VISION_EVAL_SET_PATH} not found.")
        print("Create it with ground-truth boxes -- see the format documented")
        print("at the top of this script, or run without --annotated for a smoke test.")
        sys.exit(1)

    eval_set = json.loads(VISION_EVAL_SET_PATH.read_text())
    print(f"Loaded {len(eval_set)} annotated image(s)")

    per_image = []
    total_tp = total_fp = total_fn = 0
    all_ious = []
    count_errors = []

    for item in eval_set:
        img_path = Path(config.DATA_DIR) / item["image"]
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}, skipping")
            continue

        result = agent.analyze_bytes(img_path.read_bytes(), source_label=item["image"])
        predicted = result.get("detections", [])
        ground_truth = item["ground_truth_boxes"]

        match = match_detections(predicted, ground_truth)
        precision = match["true_positives"] / max(match["true_positives"] + match["false_positives"], 1)
        recall = match["true_positives"] / max(match["true_positives"] + match["false_negatives"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        count_error = abs(len(predicted) - len(ground_truth))

        per_image.append({
            "image": item["image"],
            "predicted_count": len(predicted),
            "ground_truth_count": len(ground_truth),
            "count_error": count_error,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "mean_iou": match["mean_iou"],
            "inference_seconds": result.get("inference_seconds", 0),
        })

        total_tp += match["true_positives"]
        total_fp += match["false_positives"]
        total_fn += match["false_negatives"]
        if match["mean_iou"] > 0:
            all_ious.append(match["mean_iou"])
        count_errors.append(count_error)

        print(f"  {item['image']}: P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
              f"count_err={count_error} IoU={match['mean_iou']:.2f}")

    overall_precision = total_tp / max(total_tp + total_fp, 1)
    overall_recall = total_tp / max(total_tp + total_fn, 1)
    overall_f1 = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-9)

    return {
        "per_image": per_image,
        "overall_precision": round(overall_precision, 4),
        "overall_recall": round(overall_recall, 4),
        "overall_f1": round(overall_f1, 4),
        "mean_iou_on_matches": round(sum(all_ious) / len(all_ious), 4) if all_ious else 0.0,
        "mean_absolute_count_error": round(sum(count_errors) / max(len(count_errors), 1), 3),
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
    }


def write_report(out_dir: Path, mode: str, data) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    (out_dir / f"vision_results_{timestamp}.json").write_text(json.dumps(data, indent=2))

    lines = ["# Vision Analysis Evaluation Report", f"Mode: {mode}",
              f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    if mode == "annotated":
        lines += [
            "## Overall Metrics",
            "| Metric | Value | Good threshold |",
            "|--------|-------|-----------------|",
            f"| Precision | {data['overall_precision']} | > 0.75 |",
            f"| Recall | {data['overall_recall']} | > 0.70 |",
            f"| F1 | {data['overall_f1']} | > 0.72 |",
            f"| Mean IoU (matches) | {data['mean_iou_on_matches']} | > 0.60 |",
            f"| Mean absolute count error | {data['mean_absolute_count_error']} | < 1.0 |",
            "",
            "## Per-Image Results",
            "| Image | Pred | GT | Count err | Precision | Recall | F1 | IoU |",
            "|-------|------|----|-----------|-----------|--------|----|----|",
        ]
        for r in data["per_image"]:
            lines.append(
                f"| {r['image']} | {r['predicted_count']} | {r['ground_truth_count']} | "
                f"{r['count_error']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['mean_iou']} |"
            )
    else:
        lines += [
            "## Smoke Test Results",
            "(No ground truth provided -- reporting raw detection stats only.)",
            "",
            "| Image | Detections | Avg confidence | Latency | Classes |",
            "|-------|------------|-----------------|---------|---------|",
        ]
        for r in data:
            classes = ", ".join(r["classes_detected"]) or "none"
            conf = r["avg_confidence"] if r["avg_confidence"] is not None else "n/a"
            lines.append(f"| {r['image']} | {r['detection_count']} | {conf} | "
                        f"{r['inference_seconds']}s | {classes} |")

    report_path = out_dir / f"vision_report_{timestamp}.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", action="store_true",
                        help="Run full precision/recall/IoU eval using eval/vision_eval_set.json")
    parser.add_argument("--out", default="eval/results", help="Output directory")
    args = parser.parse_args()

    agent = VisionAnalysisAgent(
        weights=config.VISION_ANALYSIS_WEIGHTS,
        confidence=config.VISION_ANALYSIS_CONFIDENCE,
        max_detections=config.VISION_ANALYSIS_MAX_DETECTIONS,
    )
    if not agent.available:
        print("ERROR: ultralytics not installed. pip install ultralytics pillow")
        sys.exit(1)

    if args.annotated:
        data = run_annotated_eval(agent)
        print(json.dumps({k: v for k, v in data.items() if k != "per_image"}, indent=2))
        write_report(Path(args.out), "annotated", data)
    else:
        data = run_smoke_test(agent)
        write_report(Path(args.out), "smoke", data)


if __name__ == "__main__":
    main()
