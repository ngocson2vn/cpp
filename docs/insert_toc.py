#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import List, Tuple

TOC_START = "<!-- TOC START -->"
TOC_END = "<!-- TOC END -->"

HEADING_RE = re.compile(r'^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*#*\s*$')
# FENCE_RE = re.compile(r'^(```|~~~)')  # detect code fences

# Characters we keep for anchors: alnum, space, hyphen, underscore
ANCHOR_ALLOWED = re.compile(r'[^0-9a-zA-Z _-]+')


def to_github_anchor(title: str) -> str:
    """
    Convert a heading text to a GitHub-style anchor slug.
    This is a practical approximation:
      - Lowercase
      - Strip leading/trailing whitespace
      - Remove punctuation except space, hyphen, underscore
      - Replace spaces with hyphens
      - Collapse multiple hyphens
    """
    slug = title.strip().lower()
    slug = ANCHOR_ALLOWED.sub('', slug)
    slug = slug.replace(' ', '-')
    slug = re.sub(r'-{2,}', '-', slug)
    return slug


def extract_headings(lines: List[str],
                     min_level: int,
                     max_level: int,
                     exclude_patterns: List[re.Pattern]) -> List[Tuple[int, str, str]]:
    """
    Returns a list of tuples: (level, title, anchor)
    Skips headings inside fenced code blocks.
    """
    in_fence = False
    fence_tick = None  # type: str | None
    headings = []

    for line in lines:
        # Detect start/end of fenced blocks
        m_fence = re.match(r'^(```+|~~~+)\s*([a-zA-Z0-9+-_.]*)?\s*$', line)
        if m_fence:
            fence_marker = m_fence.group(1)
            if not in_fence:
                in_fence = True
                fence_tick = fence_marker
            else:
                # Close only if same tick type (``` vs ~~~)
                if fence_tick and fence_marker.startswith(fence_tick[0]):
                    in_fence = False
                    fence_tick = None
            continue

        if in_fence:
            continue

        m = HEADING_RE.match(line)
        if not m:
            continue

        level = len(m.group('hashes'))
        title = m.group('title').strip()

        if level < min_level or level > max_level:
            continue

        # Exclusion by patterns
        if any(p.search(title) for p in exclude_patterns):
            continue

        anchor = to_github_anchor(title)
        headings.append((level, title, anchor))

    return headings


def build_toc(headings: List[Tuple[int, str, str]]) -> List[str]:
    """
    Build a nested bullet list TOC from heading tuples.
    Indentation is based on relative levels, normalized to the minimum level.
    """
    if not headings:
        return []

    min_level = min(h[0] for h in headings)
    toc_lines = []
    for level, title, anchor in headings:
        indent = '  ' * (level - min_level)
        toc_lines.append(f"{indent}- [{title}](#{anchor})")
    return toc_lines


def insert_or_replace_toc(lines: List[str], toc_lines: List[str], use_marker: bool) -> List[str]:
    """
    Insert TOC at top or replace an existing marker block.
    - If use_marker: ensure a marker block exists near top, replacing any existing one.
    - Else: insert TOC as a plain block at the very start, or replace existing marker if present.
    """
    if not toc_lines:
        return lines  # nothing to insert

    toc_block = []
    if use_marker:
        toc_block.append(TOC_START)
        toc_block.extend(toc_lines)
        toc_block.append(TOC_END)
        toc_block.append("")  # blank line after block
    else:
        # Plain block with a title plus TOC. You can customize the heading here.
        toc_block.append("<!-- Auto-generated Table of Contents -->")
        toc_block.extend(toc_lines)
        toc_block.append("")

    # If there is an existing TOC marker block, replace it.
    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == TOC_START:
            start_idx = i
        if line.strip() == TOC_END and start_idx is not None and end_idx is None:
            end_idx = i
            break

    if start_idx is not None and end_idx is not None:
        # Replace existing block
        return lines[:start_idx] + [*toc_block] + lines[end_idx + 1:]

    # Else, insert near the top:
    # Place after an initial YAML front matter block if present, otherwise at file start.
    if len(lines) >= 1 and lines[0].strip() == '---':
        # Find end of front matter
        for j in range(1, len(lines)):
            if lines[j].strip() == '---':
                # Insert after front matter
                return lines[:j + 1] + [""] + toc_block + lines[j + 1:]

    # Otherwise insert at the very beginning, but skip leading blank lines
    first_content = 0
    while first_content < len(lines) and lines[first_content].strip() == '':
        first_content += 1
    return lines[:first_content] + [*toc_block] + lines[first_content:]


def main():
    parser = argparse.ArgumentParser(description="Insert or update a Table of Contents in a Markdown file.")
    parser.add_argument("markdown_file", help="Path to the Markdown file to modify in place.")
    parser.add_argument("--min-level", type=int, default=1, help="Minimum heading level to include (1-6).")
    parser.add_argument("--max-level", type=int, default=6, help="Maximum heading level to include (1-6).")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Regex pattern for headings to exclude. Can be passed multiple times.")
    parser.add_argument("--marker", action="store_true",
                        help=f"Use marker block '{TOC_START} ... {TOC_END}' for TOC placement.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the modified content to stdout instead of writing back.")
    args = parser.parse_args()

    if not (1 <= args.min_level <= 6 and 1 <= args.max_level <= 6 and args.min_level <= args.max_level):
        print("Error: min-level and max-level must be between 1 and 6, and min <= max.", file=sys.stderr)
        sys.exit(2)

    exclude_patterns = [re.compile(p) for p in args.exclude]

    path = args.markdown_file
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip('\n') for ln in f.readlines()]

    headings = extract_headings(lines, args.min_level, args.max_level, exclude_patterns)
    toc_lines = build_toc(headings)
    new_lines = insert_or_replace_toc(lines, toc_lines, args.marker)

    if args.dry_run:
        sys.stdout.write("\n".join(new_lines) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")


# python insert_toc.py inductor.md --min-level 1 --max-level 4 --exclude "Table of Contents" --marker
if __name__ == "__main__":
    main()
