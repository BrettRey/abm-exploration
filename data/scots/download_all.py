#!/usr/bin/env python3
"""
Download all SCOSYA rating data by scraping the HTML tabular data pages.

The SCOSYA API is blocked by SiteGround anti-bot protection, but the
HTML pages at /data-in-tabular-form/?id={code} serve the same data in
HTML tables and are accessible.

Usage:
    python download_all.py                # download all 258 attributes
    python download_all.py --resume       # skip already-downloaded
    python download_all.py --delay 3      # seconds between requests (default 3)
"""

import json
import csv
import time
import os
import sys
import urllib.request
from html.parser import HTMLParser

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://scotssyntaxatlas.ac.uk/data-in-tabular-form/"
DELAY = 3

resume = "--resume" in sys.argv
for i, arg in enumerate(sys.argv):
    if arg == "--delay" and i + 1 < len(sys.argv):
        DELAY = float(sys.argv[i + 1])


class TableParser(HTMLParser):
    """Extract rows from the first <table> in an HTML document."""
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.rows = []
        self.current_row = []
        self.current_cell = ""

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self.current_cell = ""

    def handle_endtag(self, tag):
        if tag == "table":
            self.in_table = False
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if self.current_row:
                self.rows.append(self.current_row)
        elif tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            self.current_row.append(self.current_cell.strip())

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data


def fetch_attribute_html(code):
    """Fetch and parse the tabular data page for one attribute."""
    url = f"{BASE_URL}?id={code}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
        "Accept": "text/html",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
            if "sgcaptcha" in html:
                return "BLOCKED"
            parser = TableParser()
            parser.feed(html)
            if len(parser.rows) < 2:
                return None
            return parser.rows
    except urllib.error.HTTPError as e:
        if e.code in (403, 202):
            return "BLOCKED"
        print(f"    HTTP {e.code}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def parse_table(rows, attr):
    """Convert parsed table rows into a list of rating dicts."""
    headers = rows[0]
    ratings = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        location = row[0]
        avg = row[1] if len(row) > 1 else ""
        young1 = row[2] if len(row) > 2 else ""
        young2 = row[3] if len(row) > 3 else ""
        old1 = row[4] if len(row) > 4 else ""
        old2 = row[5] if len(row) > 5 else ""

        # Create one row per individual rating
        for agegroup, val in [("young", young1), ("young", young2),
                               ("old", old1), ("old", old2)]:
            if val and val.strip():
                ratings.append({
                    "aid": attr["aid"],
                    "code": attr["code"],
                    "phenomenon": attr["pname"],
                    "stimulus": attr["atname"],
                    "location": location,
                    "avg_rating": avg,
                    "agegroup": agegroup,
                    "rating": val.strip(),
                })
    return ratings


def main():
    attr_file = os.path.join(DATA_DIR, "attributes.json")
    if not os.path.exists(attr_file):
        print("Missing attributes.json. Run:")
        print("  curl -s https://scotssyntaxatlas.ac.uk/api/v1/json/attributes > attributes.json")
        sys.exit(1)

    with open(attr_file) as f:
        attributes = json.load(f)

    attr_dir = os.path.join(DATA_DIR, "per_attribute")
    os.makedirs(attr_dir, exist_ok=True)

    print(f"Downloading {len(attributes)} attributes (delay={DELAY}s)...")

    blocked_count = 0
    success_count = 0
    skip_count = 0

    for i, attr in enumerate(attributes):
        code = attr["code"]
        out_file = os.path.join(attr_dir, f"{code}_{attr['aid']}.json")

        if resume and os.path.exists(out_file):
            skip_count += 1
            continue

        rows = fetch_attribute_html(code)

        if rows == "BLOCKED":
            blocked_count += 1
            print(f"  [{i+1}/{len(attributes)}] {code}: BLOCKED")
            if blocked_count >= 3:
                print(f"\n  Blocked after {success_count} successful downloads.")
                print(f"  Wait and re-run with --resume")
                break
            time.sleep(DELAY * 3)
            continue

        if rows is None:
            print(f"  [{i+1}/{len(attributes)}] {code}: no data")
            continue

        blocked_count = 0
        success_count += 1

        ratings = parse_table(rows, attr)
        with open(out_file, "w") as f:
            json.dump(ratings, f)

        print(f"  [{i+1}/{len(attributes)}] {code}: {len(ratings)} ratings from {len(rows)-1} locations")
        time.sleep(DELAY)

    if skip_count:
        print(f"\nSkipped {skip_count} already-downloaded attributes")

    # Compile all downloaded files
    print(f"\nCompiling all downloaded attributes...")
    all_ratings = []
    for attr in attributes:
        fpath = os.path.join(attr_dir, f"{attr['code']}_{attr['aid']}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            all_ratings.extend(json.load(f))

    if not all_ratings:
        print("No data to compile.")
        return

    out_json = os.path.join(DATA_DIR, "all_ratings.json")
    with open(out_json, "w") as f:
        json.dump(all_ratings, f)

    fieldnames = ["aid", "code", "phenomenon", "stimulus", "location",
                  "avg_rating", "agegroup", "rating"]
    out_csv = os.path.join(DATA_DIR, "all_ratings.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_ratings)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"\nSummary:")
    print(f"  Total individual ratings: {len(all_ratings)}")
    print(f"  Unique locations: {len(set(r['location'] for r in all_ratings))}")
    print(f"  Unique attributes: {len(set(r['code'] for r in all_ratings))}")

    from collections import Counter
    rc = Counter(r["rating"] for r in all_ratings)
    print(f"  Rating distribution:")
    for rating in sorted(rc):
        print(f"    {rating}: {rc[rating]}")


if __name__ == "__main__":
    main()
