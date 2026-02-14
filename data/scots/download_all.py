#!/usr/bin/env python3
"""
Download all SCOSYA rating data and compile into a single dataset.

The SCOSYA server (SiteGround hosting) has aggressive anti-bot protection.
If you get 403s, try:
  1. Increasing DELAY (5-10 seconds)
  2. Running from a browser session first to get cookies
  3. Running in smaller batches

Usage:
    python download_all.py              # download all
    python download_all.py --resume     # skip already-downloaded attributes
    python download_all.py --delay 5    # 5s between requests
"""

import json
import csv
import time
import os
import sys
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://scotssyntaxatlas.ac.uk/api/v1/json"
DELAY = 2  # seconds between requests

# Parse args
resume = "--resume" in sys.argv
for i, arg in enumerate(sys.argv):
    if arg == "--delay" and i + 1 < len(sys.argv):
        DELAY = float(sys.argv[i + 1])


def fetch_attribute(aid):
    """Fetch rating data for one attribute. Returns list of dicts or None."""
    url = f"{BASE_URL}/attribute/{aid}/And/all/1/12345/both/includeSpurious/point"
    req = urllib.request.Request(url, headers={
        "User-Agent": "SCOSYA-Research-Download/1.0 (academic research)",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return "BLOCKED"
        raise
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    # Load attribute list
    attr_file = os.path.join(DATA_DIR, "attributes.json")
    if not os.path.exists(attr_file):
        print("Run this first: curl -s http://scotssyntaxatlas.ac.uk/api/v1/json/attributes > attributes.json")
        sys.exit(1)

    with open(attr_file) as f:
        attributes = json.load(f)

    # Track per-attribute files for resume
    attr_dir = os.path.join(DATA_DIR, "per_attribute")
    os.makedirs(attr_dir, exist_ok=True)

    print(f"Downloading rating data for {len(attributes)} attributes (delay={DELAY}s)...")

    blocked_count = 0
    success_count = 0

    for i, attr in enumerate(attributes):
        aid = attr["aid"]
        code = attr["code"]

        out_file = os.path.join(attr_dir, f"{code}_{aid}.json")

        if resume and os.path.exists(out_file):
            success_count += 1
            continue

        data = fetch_attribute(aid)

        if data == "BLOCKED":
            blocked_count += 1
            if blocked_count >= 3:
                print(f"\n  Blocked by rate limiter after {i} requests.")
                print(f"  Wait 10-15 minutes and re-run with --resume")
                print(f"  Successfully downloaded: {success_count}")
                break
            time.sleep(DELAY * 3)  # extra backoff
            continue

        if data is None:
            continue

        blocked_count = 0  # reset on success
        success_count += 1

        # Save per-attribute
        with open(out_file, "w") as f:
            json.dump(data, f)

        n_valid = sum(1 for r in data if r.get("qid") and r.get("spurious") != "Y")
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(attributes)}] {code}: {n_valid} valid ratings")

        time.sleep(DELAY)

    # Compile all downloaded files
    print(f"\nCompiling {success_count} downloaded attributes...")
    all_ratings = []
    for attr in attributes:
        aid = attr["aid"]
        code = attr["code"]
        pname = attr["pname"]
        atname = attr["atname"]

        fpath = os.path.join(attr_dir, f"{code}_{aid}.json")
        if not os.path.exists(fpath):
            continue

        with open(fpath) as f:
            data = json.load(f)

        for row in data:
            if not row.get("qid"):
                continue
            row["aid"] = aid
            row["code_orig"] = code
            row["phenomenon"] = pname
            row["stimulus"] = atname
            all_ratings.append(row)

    if not all_ratings:
        print("No data to compile.")
        return

    # Save combined
    out_json = os.path.join(DATA_DIR, "all_ratings.json")
    with open(out_json, "w") as f:
        json.dump(all_ratings, f)

    fieldnames = ["aid", "code_orig", "phenomenon", "stimulus", "qid",
                   "display_town", "display_lat", "display_lng",
                   "cid", "rating", "spurious", "isfw", "agegroup"]
    out_csv = os.path.join(DATA_DIR, "all_ratings.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_ratings)

    valid = [r for r in all_ratings if r.get("spurious") != "Y"]
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"\nSummary:")
    print(f"  Total rows: {len(all_ratings)}")
    print(f"  Non-spurious: {len(valid)}")
    print(f"  Unique locations: {len(set(r['display_town'] for r in valid))}")
    print(f"  Unique constructions: {len(set(r['cid'] for r in valid))}")

    from collections import Counter
    rc = Counter(r["rating"] for r in valid)
    print(f"  Rating distribution:")
    for rating in sorted(rc):
        print(f"    {rating}: {rc[rating]}")


if __name__ == "__main__":
    main()
