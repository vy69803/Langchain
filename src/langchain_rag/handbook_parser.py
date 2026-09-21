"""Handbook parser and normalizer for GitLab Handbook markdown documents.

Handles YAML frontmatter extraction, Hugo shortcode sanitization, canonical URL
mapping, department/breadcrumb hierarchy extraction, and content hashing.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document

# Base URL for GitLab Handbook web pages
GITLAB_HANDBOOK_BASE_URL = "https://handbook.gitlab.com"


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and body from a markdown string.

    Args:
        content: Raw markdown text potentially starting with YAML frontmatter (---).

    Returns:
        Tuple of (metadata_dict, cleaned_body_text).
    """
    stripped = content.strip()
    if not stripped.startswith("---"):
        return {}, content

    # Find the closing frontmatter delimiter
    parts = stripped.split("---", 2)
    if len(parts) < 3:
        return {}, content

    raw_yaml = parts[1].strip()
    body = parts[2].lstrip("\r\n")

    metadata: dict[str, Any] = {}
    try:
        loaded = yaml.safe_load(raw_yaml)
        if isinstance(loaded, dict):
            metadata = loaded
    except Exception:
        # Fallback regex extraction for slightly malformed or custom Hugo tags in YAML
        for line in raw_yaml.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip().strip("'\"")
                val = val.strip().strip("'\"")
                if key and val:
                    metadata[key] = val

    return metadata, body


def clean_hugo_shortcodes(text: str) -> str:
    """Sanitize Hugo template shortcodes into clean, LLM-friendly Markdown.

    Converts alerts, panels, and notes to standard markdown blockquotes,
    converts video embeds into links, and removes orphaned Hugo syntax tags.

    Args:
        text: Markdown text with Hugo shortcodes.

    Returns:
        Sanitized markdown text.
    """
    # 1. Alerts with title: {{% alert title="Note" color="primary" %}}content{{% /alert %}}
    def _replace_titled_alert(match: re.Match[str]) -> str:
        title = match.group(1).strip()
        body = match.group(2).strip()
        return f"\n> **[{title}]** {body}\n"

    text = re.sub(
        r"\{\{[%<]\s*alert\s+title=[\"']([^\"']+)[\"'].*?[%>]\}\}(.*?)\{\{[%<]\s*/alert\s*[%>]\}\}",
        _replace_titled_alert,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 2. Generic alerts without title: {{% alert ... %}}content{{% /alert %}}
    def _replace_alert(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        return f"\n> **[Alert]** {body}\n"

    text = re.sub(
        r"\{\{[%<]\s*alert(?:\s+[^%>]*)?[%>]\}\}(.*?)\{\{[%<]\s*/alert\s*[%>]\}\}",
        _replace_alert,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 3. Panels: {{< panel header="Header" ... >}}content{{< /panel >}}
    def _replace_panel(match: re.Match[str]) -> str:
        header = match.group(1).strip()
        body = match.group(2).strip()
        # Strip internal markdown bolding from header if present
        header = header.replace("**", "").strip()
        return f"\n> **[{header}]** {body}\n"

    text = re.sub(
        r"\{\{[%<]\s*panel\s+header=[\"']([^\"']+)[\"'].*?[%>]\}\}(.*?)\{\{[%<]\s*/panel\s*[%>]\}\}",
        _replace_panel,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 4. Notes: {{% note %}}content{{% /note %}}
    def _replace_note(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        return f"\n> **Note**: {body}\n"

    text = re.sub(
        r"\{\{[%<]\s*note\s*[%>]\}\}(.*?)\{\{[%<]\s*/note\s*[%>]\}\}",
        _replace_note,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 5. YouTube video embeds: {{< youtube "VIDEO_ID" >}}
    text = re.sub(
        r"\{\{[%<]\s*youtube\s+[\"']([^\"']+)[\"'].*?[%>]\}\}",
        r"[Video Resource: https://youtube.com/watch?v=\1]",
        text,
        flags=re.IGNORECASE,
    )

    # 6. Hugo figure shortcodes: {{< figure src="URL" ... title="TITLE" ... >}}
    def _replace_figure(match: re.Match[str]) -> str:
        src = match.group(1).strip()
        return f"![Figure]({src})"

    text = re.sub(
        r"\{\{[%<]\s*figure\s+src=[\"']([^\"']+)[\"'].*?[%>]\}\}",
        _replace_figure,
        text,
        flags=re.IGNORECASE,
    )

    # 7. Strip any remaining opening/closing Hugo shortcodes: {{< ... >}} or {{% ... %}}
    text = re.sub(r"\{\{[%<].*?[%>]\}\}", "", text)

    # Clean up multiple consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compute_content_hash(text: str) -> str:
    """Calculate SHA-256 hash of text for change detection and idempotency."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_first_heading(text: str) -> str | None:
    """Extract the first markdown H1 or H2 heading from text if present."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# ") and len(line) > 2:
            return line[2:].strip()
        if line.startswith("## ") and len(line) > 3:
            return line[3:].strip()
    return None


def resolve_handbook_url(
    file_path: Path | str,
    base_url: str = GITLAB_HANDBOOK_BASE_URL,
) -> str:
    """Map a local handbook markdown file path to its canonical public web URL.

    Examples:
        'handbook/content/handbook/_index.md' -> 'https://handbook.gitlab.com/handbook/'
        'handbook/content/handbook/values/index.md' -> 'https://handbook.gitlab.com/handbook/values/'
        'handbook/content/handbook/engineering/architecture.md'
            -> 'https://handbook.gitlab.com/handbook/engineering/architecture/'
    """
    p = Path(file_path).as_posix()

    # Find where 'content/handbook' or 'handbook' starts
    marker = "content/handbook/"
    if marker in p:
        rel = p.split(marker, 1)[1]
    elif "content/" in p:
        rel = p.split("content/", 1)[1]
    elif "handbook/" in p:
        rel = p.split("handbook/", 1)[1]
    else:
        rel = Path(file_path).name

    # Normalize file stems
    if rel.endswith("/_index.md") or rel == "_index.md":
        rel = rel[:-10]
    elif rel.endswith("/index.md") or rel == "index.md":
        rel = rel[:-9]
    elif rel.endswith(".md"):
        rel = rel[:-3]

    rel = rel.strip("/")
    if rel:
        return f"{base_url.rstrip('/')}/handbook/{rel}/"
    return f"{base_url.rstrip('/')}/handbook/"


def extract_department_and_breadcrumbs(file_path: Path | str) -> tuple[str, list[str]]:
    """Extract the top-level department and breadcrumb hierarchy from path.

    Example:
        'handbook/content/handbook/engineering/development/architecture.md'
        -> department: 'engineering'
        -> breadcrumbs: ['Handbook', 'Engineering', 'Development', 'Architecture']
    """
    p = Path(file_path).as_posix()
    marker = "content/handbook/"
    if marker in p:
        subpath = p.split(marker, 1)[1]
    elif "content/" in p:
        subpath = p.split("content/", 1)[1]
    elif "handbook/" in p:
        subpath = p.split("handbook/", 1)[1]
    else:
        subpath = Path(file_path).name

    parts = [part for part in subpath.split("/") if part]
    if not parts:
        return "general", ["Handbook"]

    department = parts[0] if parts[0] not in ("_index.md", "index.md") else "general"

    # Build clean breadcrumb labels
    crumbs = ["Handbook"]
    for part in parts:
        clean = part
        if clean in ("_index.md", "index.md"):
            continue
        if clean.endswith(".md"):
            clean = clean[:-3]
        clean_label = clean.replace("-", " ").replace("_", " ").title()
        if clean_label and clean_label not in crumbs:
            crumbs.append(clean_label)

    return department, crumbs


class HandbookParser:
    """Parser and normalizer for GitLab Handbook markdown documents."""

    def __init__(self, base_url: str = GITLAB_HANDBOOK_BASE_URL) -> None:
        self.base_url = base_url

    def parse_text(
        self,
        content: str,
        file_path: Path | str = "document.md",
    ) -> Document:
        """Parse raw markdown content with frontmatter into a normalized LangChain Document.

        Args:
            content: Raw markdown text string.
            file_path: Associated filesystem path or logical identifier.

        Returns:
            LangChain Document with cleaned content and structured metadata.
        """
        raw_meta, raw_body = parse_frontmatter(content)
        cleaned_body = clean_hugo_shortcodes(raw_body)
        doc_hash = compute_content_hash(cleaned_body)

        path_obj = Path(file_path)
        department, breadcrumbs = extract_department_and_breadcrumbs(file_path)
        canonical = raw_meta.get("canonical_path")
        if canonical:
            if canonical.startswith("/"):
                canonical_url = f"{self.base_url.rstrip('/')}{canonical}"
            else:
                canonical_url = canonical
        else:
            canonical_url = resolve_handbook_url(file_path, self.base_url)

        # Determine document title
        title = (
            raw_meta.get("title")
            or extract_first_heading(cleaned_body)
            or breadcrumbs[-1]
            or path_obj.stem.replace("-", " ").replace("_", " ").title()
        )
        if isinstance(title, str):
            title = title.strip().strip("'\"")

        description = raw_meta.get("description") or raw_meta.get("decsription") or ""
        tags = raw_meta.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        metadata: dict[str, Any] = {
            "source": path_obj.as_posix(),
            "filename": path_obj.name,
            "title": str(title),
            "description": str(description),
            "department": department,
            "breadcrumbs": " > ".join(breadcrumbs),
            "breadcrumb_list": breadcrumbs,
            "url": canonical_url,
            "doc_hash": doc_hash,
            "char_count": len(cleaned_body),
            "word_count": len(cleaned_body.split()),
            "last_updated": str(raw_meta.get("last_updated", "")),
        }

        # Include any optional frontmatter properties if present
        for extra_key in ("controlled_document", "category", "stage", "group"):
            if extra_key in raw_meta:
                metadata[extra_key] = raw_meta[extra_key]

        return Document(page_content=cleaned_body, metadata=metadata)

    def parse_file(self, file_path: Path | str) -> Document:
        """Read and parse a single handbook markdown file from disk.

        Args:
            file_path: Path to the markdown file.

        Returns:
            LangChain Document with cleaned content and structured metadata.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Handbook file not found: {path}")

        content = path.read_text(encoding="utf-8", errors="ignore")
        return self.parse_text(content, file_path=path)
