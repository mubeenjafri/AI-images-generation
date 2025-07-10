#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Any, Tuple, cast, Literal
import base64

import requests
from flask import Flask, jsonify, render_template_string 
from openai import OpenAI, OpenAIError  # type: ignore
from dotenv import load_dotenv

# ------------------------ Configuration helpers ----------------------------- #

# GPT Image 1 supported sizes
ORIENTATION_TO_SIZE: Dict[str, str] = {
    "square": "1024x1024",
    "portrait": "1024x1536", 
    "landscape": "1536x1024",
    # Keep backward compatibility with old names
    "vertical": "1024x1536",
    "horizontal": "1536x1024",
}

# DALL-E 3 supported sizes (different from GPT Image 1)
DALLE3_ORIENTATION_TO_SIZE: Dict[str, str] = {
    "square": "1024x1024",
    "portrait": "1024x1792", 
    "landscape": "1792x1024",
    # Keep backward compatibility with old names
    "vertical": "1024x1792",
    "horizontal": "1792x1024",
}

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image Generation Progress</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    tr:nth-child(even){background-color: #f2f2f2;}
    .bar { height: 16px; background-color: #4CAF50; }
  </style>
</head>
<body>
  <h2>Progress</h2>
  <p>Total images: <span id="total"></span> | Completed: <span id="completed"></span></p>
  <table>
    <thead>
      <tr><th>#</th><th>Filename</th><th>Prompt</th><th>Status</th><th>Completed / Target</th></tr>
    </thead>
    <tbody id="table-body"></tbody>
  </table>
<script>
function update() {
  fetch('/progress').then(r => r.json()).then(data => {
    document.getElementById('total').innerText = data.total_images;
    document.getElementById('completed').innerText = data.completed_images;
    const body = document.getElementById('table-body');
    body.innerHTML = '';
    data.entries.forEach((e, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${idx+1}</td><td>${e.filename}</td><td>${e.prompt}</td>`+
        `<td>${e.status}</td><td>${e.done}/${e.target}</td>`;
      body.appendChild(tr);
    });
    if (data.completed_images < data.total_images) {
      setTimeout(update, 1000);
    }
  }).catch(() => setTimeout(update, 2000));
}
update();
</script>
</body>
</html>
"""


class ProgressTracker:
    """Thread-safe progress tracker for image generation."""

    def __init__(self, total_images: int):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "total_images": total_images,
            "completed_images": 0,
            "entries": [],
        }

    def add_entry(self, prompt: str, filename: str, target: int):
        with self._lock:
            self._data["entries"].append({
                "prompt": prompt,
                "filename": filename,
                "target": target,
                "done": 0,
                "status": "pending",
                "error": None,
            })

    def mark_in_progress(self, index: int):
        with self._lock:
            self._data["entries"][index]["status"] = "in_progress"

    def increment_done(self, index: int):
        with self._lock:
            entry = self._data["entries"][index]
            entry["done"] += 1
            if entry["done"] == entry["target"]:
                entry["status"] = "done"
            self._data["completed_images"] += 1

    def mark_error(self, index: int, exc: Exception):
        with self._lock:
            entry = self._data["entries"][index]
            entry["status"] = "error"
            entry["error"] = str(exc)
            # Even though errored, count toward completed to prevent hang
            entry["done"] = entry["target"]
            self._data["completed_images"] += (entry["target"] - entry["done"])

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._data.copy()


# ----------------------------- Flask server --------------------------------- #

def make_app(tracker: ProgressTracker) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/progress")
    def progress():
        return jsonify(tracker.snapshot())

    return app


# ---------------------------- Image generator ------------------------------ #

def generate_images(
    client: OpenAI,
    tracker: ProgressTracker,
    rows_to_process: List[Tuple[int, Dict[str, str]]],
    all_rows: List[Dict[str, str]],
    out_dir: Path,
    n: int,
    model: str,
):
    """Generate images for each row in rows_to_process.

    rows_to_process: list of (original_index, row_dict) where status != 'done'.
    all_rows: reference to full list for status updates.
    model: either "gpt-image-1" or "dall-e-3"
    """

    for proc_idx, (orig_idx, row) in enumerate(rows_to_process):
        prompt = row.get("prompt", "").strip()
        base_filename = row.get("filename", f"image_{proc_idx:03}").strip()
        orientation = row.get("orientation", "square").strip().lower()
        
        # Choose size mapping based on model
        if model == "gpt-image-1":
            size = ORIENTATION_TO_SIZE.get(orientation, ORIENTATION_TO_SIZE["square"])
        else:
            size = DALLE3_ORIENTATION_TO_SIZE.get(orientation, DALLE3_ORIENTATION_TO_SIZE["square"])

        tracker.mark_in_progress(proc_idx)

        try:
            if model == "gpt-image-1":
                # GPT Image 1 configuration
                size_literal = cast(
                    Literal[
                        "1024x1024",
                        "1024x1536",
                        "1536x1024",
                    ],
                    size,
                )
                
                # Generate n images by making n separate API calls
                for i in range(1, n + 1):
                    response = client.images.generate( 
                        model="gpt-image-1",
                        prompt=prompt,
                        size=size_literal,
                        quality="high",  # Use high quality for best results
                        output_format="png",  # Ensure PNG format
                    )
                    
                    if not response.data or not response.data[0].b64_json:
                        print(f"Warning: No image data returned for {base_filename}_{i}")
                        continue
                        
                    # GPT Image 1 returns base64 encoded image data
                    image_base64 = response.data[0].b64_json
                    file_stem = f"{base_filename}_{i}" if n > 1 else base_filename
                    file_path = out_dir / f"{file_stem}.png"
                    
                    # Save the base64 image data
                    save_base64_image(image_base64, file_path)
                    tracker.increment_done(proc_idx)
                    
            elif model == "dall-e-3":
                # DALL-E 3 configuration
                size_literal = cast(
                    Literal[
                        "1024x1024",
                        "1024x1792",
                        "1792x1024",
                    ],
                    size,
                )
                
                # DALL-E 3 only supports n=1, so we need to make multiple calls for n > 1
                for i in range(1, n + 1):
                    response = client.images.generate( 
                        model="dall-e-3",
                        prompt=prompt,
                        n=1,  # DALL-E 3 only supports n=1
                        size=size_literal,
                    )
                    
                    if not response.data or not response.data[0].url:
                        print(f"Warning: No image URL returned for {base_filename}_{i}")
                        continue
                        
                    url = response.data[0].url
                    file_stem = f"{base_filename}_{i}" if n > 1 else base_filename
                    file_path = out_dir / f"{file_stem}.png"
                    download_and_save(url, file_path)
                    tracker.increment_done(proc_idx)

            # Mark as done in the master rows list
            all_rows[orig_idx]["status"] = "done"

        except OpenAIError as oe:
            tracker.mark_error(proc_idx, oe)
            all_rows[orig_idx]["status"] = "error"
            print(f"OpenAI API error for prompt #{proc_idx+1}: {oe}", file=sys.stderr)
        except Exception as exc:
            tracker.mark_error(proc_idx, exc)
            all_rows[orig_idx]["status"] = "error"
            print(f"Unexpected error for prompt #{proc_idx+1}: {exc}", file=sys.stderr)


def save_base64_image(base64_data: str, path: Path):
    """Save base64 encoded image data to path, handling existing files safely."""
    # If file exists, create a unique name
    if path.exists():
        base_stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        
        while path.exists():
            new_stem = f"{base_stem}_conflict_{counter}"
            path = parent / f"{new_stem}{suffix}"
            counter += 1
        
        print(f"Warning: File conflict resolved. Saved as: {path.name}")
    
    # Decode and save the base64 image data
    image_bytes = base64.b64decode(base64_data)
    path.write_bytes(image_bytes)


def download_and_save(url: str, path: Path):
    """Download an image from URL and save to path, handling existing files safely."""
    # If file exists, create a unique name
    if path.exists():
        base_stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        
        while path.exists():
            new_stem = f"{base_stem}_conflict_{counter}"
            path = parent / f"{new_stem}{suffix}"
            counter += 1
        
        print(f"Warning: File conflict resolved. Saved as: {path.name}")
    
    resp = requests.get(url)
    resp.raise_for_status()
    path.write_bytes(resp.content)


# ------------------------------- CLI entry ---------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images via OpenAI API from a CSV file.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file (with headers: prompt, filename, orientation).")
    parser.add_argument("--out", default="output", help="Destination folder for PNG images (created if missing).")
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate per prompt (1-10).")
    parser.add_argument("--port", type=int, default=5000, help="Port for the local progress web server.")
    parser.add_argument("--model", choices=["gpt-image-1", "dall-e-3"], default="dall-e-3", help="Image generation model to use.")
    return parser.parse_args()


def read_csv(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV and return rows plus original header order."""
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        original_headers: List[str] = list(reader.fieldnames) if reader.fieldnames else []
        rows = list(reader)

    # Ensure status column exists in rows (blank if missing)
    for row in rows:
        row.setdefault("status", "")

    return rows, original_headers


def write_csv(csv_path: Path, rows: List[Dict[str, str]], headers: List[str]):
    """Write rows back to CSV preserving header order and including status column."""
    # Ensure 'status' is in headers (append if not present)
    if "status" not in headers:
        headers.append("status")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_free_port(start_port: int = 5000) -> int:
    """Find the first available port starting from start_port."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free ports found starting from {start_port}")


def main():
    args = parse_args()

    if not (1 <= args.n <= 10):
        print("--n must be between 1 and 10", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows, original_headers = read_csv(csv_path)

    if not rows:
        print("CSV appears empty.", file=sys.stderr)
        sys.exit(1)

    # Filter rows that still need processing
    rows_to_process: List[Tuple[int, Dict[str, str]]] = [
        (idx, r) for idx, r in enumerate(rows) if (r.get("status") or "").lower() != "done"
    ]

    if not rows_to_process:
        print("Nothing to do: all rows already marked as done.")
        sys.exit(0)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(rows_to_process) * args.n
    tracker = ProgressTracker(total_images=total_images)
    for _, row in rows_to_process:
        tracker.add_entry(row.get("prompt", ""), row.get("filename", ""), args.n)

    load_dotenv()

    if "OPENAI_API_KEY" not in os.environ:
        print("OPENAI_API_KEY not found. Add it to a .env file or set it in your environment.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI()

    # Find available port
    try:
        available_port = find_free_port(args.port)
        if available_port != args.port:
            print(f"Port {args.port} not available, using port {available_port}")
    except RuntimeError as e:
        print(f"Error finding available port: {e}", file=sys.stderr)
        sys.exit(1)

    # Launch Flask server in background
    app = make_app(tracker)
    server_thread = threading.Thread(target=lambda: app.run(port=available_port, debug=False, use_reloader=False), daemon=True)
    server_thread.start()

    # Give server a moment to start
    time.sleep(1)
    
    progress_url = f"http://localhost:{available_port}/"
    print(f"Progress page available at {progress_url}")
    print(f"Using model: {args.model}")
    
    # Auto-open browser
    try:
        webbrowser.open(progress_url)
        print("Browser opened automatically.")
    except Exception:
        print("Could not open browser automatically. Please open the URL manually.")

    # Generate images (blocking)
    generate_images(client, tracker, rows_to_process, rows, out_dir, args.n, args.model)

    # After processing, write CSV back with updated statuses
    write_csv(csv_path, rows, original_headers)

    # Allow user to view final state; keep server alive for a short grace period
    print("All tasks completed. You may close the script when done viewing progress.")
    try:
        while server_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting on user interrupt.")


if __name__ == "__main__":
    main() 