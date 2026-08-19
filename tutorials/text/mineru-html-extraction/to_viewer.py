#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Publish a MinerU-HTML run into the layout the Atlas `main-content` viewer reads.

The viewer wants a document cut into identified elements, one run's label per element,
and the markdown that run produced. This pipeline already has all three — they are just
spelled differently:

    viewer                     MinerU
    element id                 `_item_id="N"` in `_mineru_map_html`
    marked HTML (left pane)    `_mineru_map_html`, with the attribute renamed
    labels {id: main|other}    `parse_compact_response(_mineru_response)`
    run markdown (right pane)  the `text` column

So this translates rather than recomputes. Nothing here re-reads the original HTML, and
no label is derived a second time: a viewer that recomputed either would eventually
disagree with the run it claims to display, and a reader could not tell which was lying.

    python to_viewer.py --run out-5-opus --out $LUSTRE_HOME/scratch/main-content/mineru-5 \
        --run-id opus-5__explicit-compact-v2 --model us/aws/anthropic/eccn-claude-opus-5

Needs the run to have been written with `--keep-internal-fields`; without them the map
HTML and the raw answer are gone and only the markdown survives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from lxml import html as lxml_html

from nemo_curator.stages.text.html_extraction.mineru_utils import (
    COVERAGE_FIELD,
    ITEM_ID_ATTR,
    MAP_HTML_FIELD,
    N_ITEMS_FIELD,
    RESPONSE_FIELD,
    STATUS_FIELD,
    label_coverage,
    parse_compact_response,
)

VIEWER_ATTR = "data-mc-id"
_WS = re.compile(r"\s+")


def collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def elements_from_map_html(map_html: str) -> tuple[list[dict[str, Any]], str]:
    """Every `_item_id` element, and the same document marked for the viewer.

    The id is MinerU's, unchanged — that is the whole point, since the labels are keyed
    on it. Only the attribute *name* changes, so the viewer's frame can find them
    without the viewer learning what `_item_id` is.
    """
    if not map_html or not map_html.strip():
        return [], ""

    tree = lxml_html.fromstring(map_html)
    elements: list[dict[str, Any]] = []

    for node in tree.xpath(f"//*[@{ITEM_ID_ATTR}]"):
        item_id = str(node.get(ITEM_ID_ATTR))
        text = collapse(" ".join(node.itertext()))
        node.set(VIEWER_ATTR, item_id)
        elements.append(
            {
                "id": item_id,
                "tag": str(node.tag).lower(),
                "cls": collapse(str(node.get("class") or "")),
                "native_id": str(node.get("id") or ""),
                "path": tree.getroottree().getpath(node),
                "depth": str(tree.getroottree().getpath(node)).count("/"),
                "leaf": False,
                "fragment": 0,
                "chars": len(text),
                "text": text,
                # The viewer renders this in its element list. MinerU converts the whole
                # pruned document at once rather than per element, so there is no
                # per-element markdown to show — the text is what there is, and claiming
                # otherwise would be inventing one.
                "md": text,
                "section": "",
                "text_sha1": hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest(),
            }
        )

    elements.sort(key=lambda e: int(e["id"]) if e["id"].isdigit() else 0)
    return elements, lxml_html.tostring(tree, encoding="unicode")


def write_jsonl(directory: Path, records: list[dict[str, Any]], head: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    with (directory / "docs.jsonl").open("wb") as handle:
        for record in records:
            offset = handle.tell()
            line = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
            handle.write(line)
            entry = dict(record.pop("_index", {}))
            entry.update({"id": record["id"], "offset": offset, "length": len(line)})
            entries.append(entry)
    (directory / "index.json").write_text(
        json.dumps({**head, "count": len(entries), "docs": entries}, indent=2, ensure_ascii=False)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="A MinerU output directory")
    ap.add_argument("--out", required=True, help="Viewer root; elements/ and runs/ go here")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="", help="Served model id, recorded on the run")
    ap.add_argument("--prompt-file", default="", help="Prompt used, recorded verbatim on the run")
    ap.add_argument("--dataset", default="arxiv/mineru-html/sample")
    ap.add_argument("--id-field", default="url")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--elements", action="store_true", help="Also write elements/ (once per corpus)")
    args = ap.parse_args()

    files = sorted(Path(args.run).glob("**/*.parquet"))
    if not files:
        msg = f"no parquet under {args.run}"
        raise SystemExit(msg)
    frame = pq.read_table(files[0], use_threads=False).to_pandas()
    for extra in files[1:]:
        import pandas as pd

        frame = pd.concat([frame, pq.read_table(extra, use_threads=False).to_pandas()], ignore_index=True)

    for column in (MAP_HTML_FIELD, RESPONSE_FIELD):
        if column not in frame.columns:
            msg = f"{args.run} has no {column}; re-run the pipeline with --keep-internal-fields"
            raise SystemExit(msg)

    out = Path(args.out)
    element_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        doc_id = str(row[args.id_field])
        elements, marked = elements_from_map_html(row[MAP_HTML_FIELD] or "")
        labels = parse_compact_response(row[RESPONSE_FIELD] or "")
        n_items = int(row.get(N_ITEMS_FIELD, len(elements)) or 0)
        coverage = label_coverage(labels, n_items)
        markdown = str(row[args.text_field] or "")
        title = next((e["text"] for e in elements if e["tag"] in ("h1", "title") and e["text"]), "")
        kept = sum(e["chars"] for e in elements if labels.get(e["id"]) == "main")
        total = sum(e["chars"] for e in elements)

        element_records.append(
            {
                "id": doc_id,
                "url": doc_id,
                "title": title,
                "source": {"status": str(row.get(STATUS_FIELD, "")), "coverage": float(row.get(COVERAGE_FIELD, 1.0))},
                "status": "ok" if elements else "empty_html",
                "nElements": len(elements),
                "chars": total,
                "elements": elements,
                "markedHtml": marked,
                "_index": {
                    "title": title,
                    "url": doc_id,
                    "status": str(row.get(STATUS_FIELD, "")),
                    "nElements": len(elements),
                    "chars": total,
                    "tier": None,
                    "era": None,
                },
            }
        )
        run_records.append(
            {
                "id": doc_id,
                "labels": labels,
                "markdown": markdown,
                "rawResponses": [str(row[RESPONSE_FIELD] or "")],
                "status": str(row.get(STATUS_FIELD, "")),
                "error": ""
                if coverage["coverage"] >= 1.0
                else f"{coverage['labelled']} of {n_items} elements labelled",
                "nLabelled": int(coverage["labelled"]),
                "nRequests": 1,
                "nUnknownIds": int(coverage["unknown"]),
                "keptChars": kept,
                "totalChars": total,
                "_index": {
                    "status": str(row.get(STATUS_FIELD, "")),
                    "nLabelled": int(coverage["labelled"]),
                    "nMain": sum(1 for v in labels.values() if v == "main"),
                    "keptChars": kept,
                    "totalChars": total,
                    "markdownChars": len(markdown),
                },
            }
        )

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.elements:
        write_jsonl(
            out / "elements",
            element_records,
            {
                "dataset": args.dataset,
                "kind": "main-content-elements",
                "formatVersion": 1,
                "createdAt": stamp,
                "tilingVerified": False,
                "selection": {"rule": f"whatever {args.run} contains", "seed": 0},
                "command": f"to_viewer.py --run {args.run}",
                "gitCommit": None,
                "gitDirty": None,
            },
        )

    run_dir = out / "runs" / args.run_id
    write_jsonl(
        run_dir,
        run_records,
        {
            "dataset": f"{args.dataset}/{args.run_id}",
            "kind": "main-content-run",
            "formatVersion": 1,
            "runId": args.run_id,
            "createdAt": stamp,
            "elementsDataset": args.dataset,
        },
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "runId": args.run_id,
                "modelId": args.run_id.split("__")[0],
                "servedModel": args.model,
                "promptId": args.run_id.split("__")[-1],
                "promptSha256": hashlib.sha256(Path(args.prompt_file).read_bytes()).hexdigest()
                if args.prompt_file and Path(args.prompt_file).is_file()
                else "",
                "promptSystem": "",
                "promptUser": Path(args.prompt_file).read_text()
                if args.prompt_file and Path(args.prompt_file).is_file()
                else "",
                "elementsDataset": args.dataset,
                "createdAt": stamp,
                "command": f"to_viewer.py --run {args.run} --run-id {args.run_id}",
                "gitCommit": None,
                "gitDirty": None,
            },
            indent=2,
        )
    )
    print(f"  {len(run_records)} document(s) -> {out}")
    print(f"  elements: {'written' if args.elements else 'skipped (--elements to write)'}")
    print(f"  run:      {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
