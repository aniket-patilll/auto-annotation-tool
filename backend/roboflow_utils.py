"""
Roboflow integration helpers — uses plain `requests` only (no cv2/inference_sdk).

Supported operations:
  validate_roboflow     – confirm API key + workspace work
  list_roboflow_projects – list projects in workspace
  autolabel_roboflow    – call Roboflow hosted inference per image → YOLO .txt files
  upload_job_frames     – push filtered frames to a Roboflow project (optional, for dataset building)
"""

from __future__ import annotations
import base64
import json
from pathlib import Path
from typing import List

import requests


# ── Connection helpers ─────────────────────────────────────────────────────────

def validate_roboflow(api_key: str, workspace: str) -> dict:
    """Check that the API key is valid and the workspace exists."""
    try:
        # Step 1: Check the API key is valid
        url = f"https://api.roboflow.com/?api_key={api_key}"
        r = requests.get(url, timeout=10)
        try:
            data = r.json()
        except Exception:
            return {"ok": False, "error": f"Unexpected response from Roboflow (HTTP {r.status_code})"}

        if not isinstance(data, dict):
            return {"ok": False, "error": f"Unexpected response type: {type(data).__name__}"}

        if not r.ok:
            msg = data.get("message") if isinstance(data, dict) else str(data)
            return {"ok": False, "error": msg or "Invalid API key"}

        # Step 2: If workspace provided, fetch its project list
        if not workspace:
            return {
                "ok": True,
                "workspace_name": "(no workspace set)",
                "project_count": 0,
                "projects": [],
                "note": "API key valid. Enter your workspace slug in Settings to browse projects.",
            }

        ws_url = f"https://api.roboflow.com/{workspace}?api_key={api_key}"
        wr = requests.get(ws_url, timeout=10)
        try:
            wd = wr.json()
        except Exception:
            return {"ok": False, "error": f"Could not parse workspace response (HTTP {wr.status_code})"}

        if not isinstance(wd, dict):
            return {"ok": False, "error": f"Unexpected workspace response: {str(wd)[:100]}"}

        if not wr.ok:
            msg = wd.get("message", f"Workspace '{workspace}' not found")
            return {"ok": False, "error": msg}

        ws_info = wd.get("workspace", {})
        raw_projects = ws_info.get("projects", [])
        projects = [p.get("id", "") for p in raw_projects if isinstance(p, dict)]
        return {
            "ok": True,
            "workspace_name": ws_info.get("name", workspace),
            "project_count": len(projects),
            "projects": projects[:30],
        }
    except requests.RequestException as e:
        return {"ok": False, "error": f"Network error: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def list_roboflow_projects(api_key: str, workspace: str) -> List[dict]:
    """Return list of projects in the workspace."""
    url = f"https://api.roboflow.com/{workspace}?api_key={api_key}"
    r = requests.get(url, timeout=10)
    if not r.ok:
        raise RuntimeError(f"Roboflow API error {r.status_code}: {r.text[:200]}")
    data = r.json()
    projects = data.get("workspace", {}).get("projects", [])
    result = []
    for p in projects:
        result.append({
            "id":   p.get("id", ""),
            "name": p.get("name", p.get("id", "")),
            "type": p.get("type", ""),
            "versions": p.get("versions", 0),
        })
    return result


def get_roboflow_model_versions(api_key: str, workspace: str, project_id: str) -> List[dict]:
    """Return all trained versions for a project."""
    url = f"https://api.roboflow.com/{workspace}/{project_id}?api_key={api_key}"
    r = requests.get(url, timeout=10)
    if not r.ok:
        raise RuntimeError(f"Roboflow API error {r.status_code}: {r.text[:200]}")
    data = r.json()
    versions = data.get("project", {}).get("versions", [])
    result = []
    for v in versions:
        result.append({
            "id":      v.get("id", ""),
            "version": v.get("version", "?"),
            "trained": v.get("model") is not None,
            "model_id": f"{project_id}/{v.get('version', '')}",
        })
    return result


# ── Hosted inference auto-labeler ──────────────────────────────────────────────

def autolabel_roboflow(
    image_dir: str,
    label_dir: str,
    model_id: str,         # e.g. "my-project/3"
    api_key: str,
    conf_threshold: float = 0.35,
) -> tuple[dict, list]:
    """
    Use Roboflow hosted inference API to auto-label images.
    Writes YOLO-format .txt label files to label_dir.
    Returns (stats_dict, class_names_list).

    Uses plain requests + base64 encoding — no cv2 or inference_sdk required.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Roboflow serverless inference endpoint:
    # POST https://serverless.roboflow.com/{model_id}?api_key={key}
    # Body: image as base64 or raw bytes, Content-Type: application/x-www-form-urlencoded
    infer_url = f"https://serverless.roboflow.com/{model_id}"

    image_paths = sorted(
        list(image_dir.glob("*.jpg"))  + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png")) + list(image_dir.glob("*.JPG"))
        + list(image_dir.glob("*.JPEG")) + list(image_dir.glob("*.PNG"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images in {image_dir}")

    stats: dict = {"total": len(image_paths), "labelled": 0, "empty": 0, "error": 0}
    all_class_names: set[str] = set()

    print(f"\n  [Roboflow] model={model_id}  conf>={conf_threshold}")
    print(f"  [Roboflow] Auto-labelling {len(image_paths)} images via cloud API...")

    for i, img_path in enumerate(image_paths):
        try:
            # Encode image as base64
            img_bytes = img_path.read_bytes()
            b64 = base64.b64encode(img_bytes).decode("utf-8")

            # Retry up to 3 times for network flakiness / DNS failures
            resp = None
            last_exc = None
            for attempt in range(3):
                try:
                    resp = requests.post(
                        infer_url,
                        params={"api_key": api_key, "confidence": int(conf_threshold * 100)},
                        data=b64,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        timeout=30,
                    )
                    last_exc = None
                    break
                except (requests.ConnectionError, requests.Timeout) as e:
                    last_exc = e
                    if attempt < 2:
                        import time; time.sleep(2)

            if last_exc is not None:
                print(f"  [Roboflow] Network error on {img_path.name} (3 attempts): {last_exc}")
                stats["error"] += 1
                continue

            if not resp.ok:
                err_text = resp.text[:200]
                print(f"  [Roboflow] HTTP {resp.status_code} on {img_path.name}: {err_text}")
                if resp.status_code == 404:
                    print(f"  [Roboflow] 404 TIP: Use 'project-slug/version' (e.g. slcm-hawg5/1), NOT 'workspace/project'")
                stats["error"] += 1
                continue

            result = resp.json()
            predictions = result.get("predictions", [])

            # Collect class names from this batch
            for pred in predictions:
                class_name = pred.get("class", "object")
                all_class_names.add(class_name)

            if not predictions:
                stats["empty"] += 1
                (label_dir / img_path.with_suffix(".txt").name).write_text("")
                continue

            # Stable class name → int mapping (alphabetical order)
            sorted_classes = sorted(all_class_names)
            cls_idx_map = {n: idx for idx, n in enumerate(sorted_classes)}

            # Image dimensions from result
            img_w = result.get("image", {}).get("width", 640)
            img_h = result.get("image", {}).get("height", 640)

            label_lines = []
            for pred in predictions:
                class_name = pred.get("class", "object")
                conf_val   = pred.get("confidence", 1.0)
                if conf_val < conf_threshold:
                    continue
                cls_id = cls_idx_map.get(class_name, 0)
                # Roboflow returns x, y as center coords in pixels
                cx = pred.get("x", 0) / img_w
                cy = pred.get("y", 0) / img_h
                bw = pred.get("width",  0) / img_w
                bh = pred.get("height", 0) / img_h
                label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf_val:.4f}")

            (label_dir / img_path.with_suffix(".txt").name).write_text("\n".join(label_lines))
            stats["labelled"] += 1

            if (i + 1) % 10 == 0:
                print(f"  [Roboflow] {i + 1}/{len(image_paths)} done...")

        except requests.Timeout:
            print(f"  [Roboflow] Timeout on {img_path.name}")
            stats["error"] += 1
        except Exception as exc:
            print(f"  [Roboflow] Error on {img_path.name}: {exc}")
            stats["error"] += 1

    class_names = sorted(all_class_names)
    print(f"\n  [Roboflow] Done: labelled={stats['labelled']} empty={stats['empty']} errors={stats['error']}")
    print(f"  [Roboflow] Classes detected: {class_names}")
    return stats, class_names


# ── Frame upload (optional — for building Roboflow datasets) ──────────────────

def upload_job_frames(
    job_output_dir: str,
    api_key: str,
    workspace: str,
    project_slug: str,
    batch_name: str = "",
) -> dict:
    """
    Upload all frames from 2_filtered_frames/ to a Roboflow project via REST API.
    Returns {uploaded, skipped, errors, total}.
    """
    frames_dir = Path(job_output_dir) / "2_filtered_frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"No filtered frames dir: {frames_dir}")

    image_paths = sorted(
        list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.jpeg"))
        + list(frames_dir.glob("*.png"))
    )
    if not image_paths:
        raise ValueError(f"No images found in {frames_dir}")

    stats = {"uploaded": 0, "skipped": 0, "errors": 0, "total": len(image_paths)}
    batch = batch_name or Path(job_output_dir).parent.name
    upload_url = f"https://api.roboflow.com/dataset/{project_slug}/upload?api_key={api_key}&batch={batch}"

    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                resp = requests.post(
                    upload_url,
                    files={"file": (img_path.name, f, "image/jpeg")},
                    timeout=30,
                )
            if resp.ok:
                result = resp.json()
                if result.get("duplicate"):
                    stats["skipped"] += 1
                else:
                    stats["uploaded"] += 1
            else:
                print(f"  [Roboflow Upload] HTTP {resp.status_code} on {img_path.name}: {resp.text[:100]}")
                stats["errors"] += 1
        except Exception as exc:
            print(f"  [Roboflow Upload] Error on {img_path.name}: {exc}")
            stats["errors"] += 1

    return stats
