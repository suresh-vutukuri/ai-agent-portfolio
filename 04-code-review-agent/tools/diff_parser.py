"""Parses unified diffs into structured per-file added/removed line data."""

from __future__ import annotations

from dataclasses import dataclass, field

from unidiff import PatchSet


@dataclass
class ChangedLine:
    """One added or removed line from a diff hunk.

    Attributes:
        line_number: The line's number in the new file (for added lines) or
            the old file (for removed lines).
        content: The line's text, without the diff's leading +/- marker.
    """

    line_number: int
    content: str


@dataclass
class ChangedFile:
    """One file's changes from a unified diff.

    Attributes:
        filename: The file's path (its new path, for renames).
        added_lines: Lines added, in diff order.
        removed_lines: Lines removed, in diff order.
    """

    filename: str
    added_lines: list[ChangedLine] = field(default_factory=list)
    removed_lines: list[ChangedLine] = field(default_factory=list)


def parse_diff(diff_text: str) -> list[ChangedFile]:
    """Parse a unified diff into structured per-file line changes.

    Args:
        diff_text: A unified diff, e.g. the output of `git diff` or a
            GitHub pull request diff.

    Returns:
        One ChangedFile per file touched by the diff, in diff order. Files
        with no textual hunks (e.g. pure renames, binary files) get empty
        added_lines/removed_lines.

    Raises:
        unidiff.errors.UnidiffParseError: If diff_text isn't a valid
            unified diff.
    """
    patch_set = PatchSet(diff_text)

    changed_files: list[ChangedFile] = []
    for patched_file in patch_set:
        added_lines: list[ChangedLine] = []
        removed_lines: list[ChangedLine] = []

        for hunk in patched_file:
            for line in hunk:
                if line.is_added:
                    added_lines.append(ChangedLine(line_number=line.target_line_no, content=line.value.rstrip("\n")))
                elif line.is_removed:
                    removed_lines.append(ChangedLine(line_number=line.source_line_no, content=line.value.rstrip("\n")))

        changed_files.append(
            ChangedFile(filename=patched_file.path, added_lines=added_lines, removed_lines=removed_lines)
        )

    return changed_files
