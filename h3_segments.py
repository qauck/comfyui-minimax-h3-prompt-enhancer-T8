"""Deterministic timeline segmentation of one finished H3 prompt.

CPU-only by design: no provider, ComfyUI, torch, model or network imports are
allowed in this module. The input is an already complete H3 prompt (Base three
fields or Ref2VA/Hybrid six fields). The output is N independent H3 prompts, each
covering one contiguous group of the authored timeline.

Contract:
- shots are grouped by their authored cut times; a shot is never split;
- `[Shot 1]` may carry no cut time, or `At 00:00.000,` which names the same instant
  and is treated as equivalent to carrying none; a first shot opening at any other
  time is rejected because the part before it would belong to no segment;
- inside every segment the timeline restarts at [Shot 1] with no timestamp, and
  every later shot keeps the exact interval it had in the source timeline;
- the analysis sections and reference labels are carried into each segment so every
  segment is a complete prompt, with [Shot N] references renumbered to that
  segment's own numbering;
- speaker identifiers are renumbered per segment in the order speakers produce an
  actual vocal event in that segment;
- nothing is invented. Dialogue, lyrics and visible text survive byte-for-byte;
  only [Shot N] labels, cut times, keyframe alignment lines and (Sx) identifiers
  are rewritten.
- segments_preset_json hands the same segment prompts to a minimax-h3-timeline-preset
  document. It adds no prompt text: its frame numbers are pseudo 24 fps positions,
  ten seconds per segment with one second of overlap into the next one.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable

try:
    from .h3_hybrid import HYBRID, is_reference_task
    from .h3_quality import (
        BASE_FIELDS,
        REF_FIELDS,
        CUT_RE,
        LITERAL_RE,
        SHOT_RE,
        SPEAKER_RE,
        alignment_sentence,
        check_h3,
        mask_literals,
        repair_shot_timecodes,
        sections,
    )
except ImportError:  # plain module import path inside ComfyUI
    from h3_hybrid import HYBRID, is_reference_task
    from h3_quality import (
        BASE_FIELDS,
        REF_FIELDS,
        CUT_RE,
        LITERAL_RE,
        SHOT_RE,
        SPEAKER_RE,
        alignment_sentence,
        check_h3,
        mask_literals,
        repair_shot_timecodes,
        sections,
    )

SEGMENT_SCHEMA = "t8-h3-segment-split/v1"
MAX_SEGMENTS = 10
MAX_TOTAL_SECONDS = 3600.0
# The preset shape is the one in ComfyUI/output/MiniMaxH3_Presets/MiniMaxH3_Preset/preset.json.
# Its fields are carried over as they are written there, except the canvas, which is written at
# 768x768 as this node's own default. Its frame numbers are not real footage here, so they are
# pseudo values at the preset's own 24 fps: ten seconds per segment, each next segment starting
# one second before the previous one ends.
PRESET_FORMAT = "minimax-h3-timeline-preset"
PRESET_SCHEMA_VERSION = 1
PRESET_NAME = "MiniMaxH3_Preset"
PRESET_TYPE = "complete"
PRESET_CANVAS = {"width": 768, "height": 768}
PRESET_SAMPLING = {"second_pass": True, "high_resolution_steps": 2}
PRESET_TIMELINE_VERSION = 10
PRESET_FPS = 24
PRESET_SEGMENT_SECONDS = 10.0
PRESET_SEGMENT_FRAMES = int(PRESET_SEGMENT_SECONDS * PRESET_FPS)
PRESET_OVERLAP_FRAMES = PRESET_FPS
SPLIT_EVEN = "平均分段（按分段数） / Even"
SPLIT_SHOT_ALIGNED = "镜头对齐分段（不切镜头） / Shot-aligned"
SPLIT_MODE_OPTIONS = [SPLIT_EVEN, SPLIT_SHOT_ALIGNED]
SPLIT_MODE_ALIASES = {
    "even": SPLIT_EVEN,
    "average": SPLIT_EVEN,
    "shot_aligned": SPLIT_SHOT_ALIGNED,
    "shot-aligned": SPLIT_SHOT_ALIGNED,
    "snap": SPLIT_SHOT_ALIGNED,
}
TASK_MODES = ("T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA")
KEYFRAME_MODES = ("I2VA", "FL2VA", "L2VA")
BODY_FIELD = "detailed_description"
BASE_BODY_FIELD = BASE_FIELDS[0]
SCENE_TRANS_RE = re.compile(r"<\s*scenetrans\b", re.I)
# The located-comma repair in the quality module intentionally leaves a [Shot 1] cut clause
# alone because the authored protocol has none, so segmentation reads the opening clause here,
# accepting either comma form for the instant that is equivalent to having no cut time at all.
FIRST_SHOT_CUT_RE = re.compile(r"[ \t\r\n]*At[ \t]+(\d{2}):(\d{2})\.(\d{3})(?![\d.])[，,]", re.I)
SHOT_TOKEN_RE = re.compile(r"\[\s*Shot\s*([1-9]\d*)\s*\]", re.I)
# Both forms of a shot reference are matched in one scan, so a reference renamed to
# this segment's numbering is never rewritten a second time. Groups: 1 = the
# whitespace a clause hangs on, 2 = clause content, 3 = a bare reference number.
REFERENCE_SCAN_RE = re.compile(r"(?P<lead>[ \t]*)\(([^()]*)\)|" + SHOT_TOKEN_RE.pattern, re.I)
# A partial keyframe alignment is a documented split consequence, not a defect the
# segment can fix on its own; it is reported as a note instead of a protocol issue.
KEYFRAME_ALIGNMENT_CODES = {"h3_alignment"}


class SegmentSplitError(ValueError):
    """The finished prompt cannot be segmented without inventing content."""


@dataclass(frozen=True)
class _Shot:
    index: int
    start_ms: int
    text_start: int
    text_end: int
    # True when an At 00:00.000 opening clause was consumed here instead of being copied
    # into the segment, so the segment can keep the space that clause hung on.
    clause_dropped: bool = False


def _error(message: str) -> SegmentSplitError:
    return SegmentSplitError(message)


def normalize_split_mode(value: Any = SPLIT_EVEN) -> str:
    if value is None or value == "":
        return SPLIT_EVEN
    if isinstance(value, str) and value in SPLIT_MODE_OPTIONS:
        return value
    alias = SPLIT_MODE_ALIASES.get(str(value).strip().casefold())
    if alias is None:
        raise _error("Unknown segment split mode; select Even or Shot-aligned.")
    return alias


def _milliseconds(seconds: Any, name: str) -> int:
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        raise _error(f"{name} must be a positive number of seconds.") from None
    if value <= 0:
        raise _error(f"{name} must be a positive number of seconds.")
    return int(round(value * 1000))


def _clock(total_ms: int) -> str:
    minutes, rest = divmod(int(total_ms), 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{minutes % 100:02d}:{seconds:02d}.{millis:03d}"


def _seconds_text(total_ms: int) -> float:
    return round(total_ms / 1000.0, 3)


def _required_fields(task_mode: str) -> tuple[str, ...]:
    return REF_FIELDS if is_reference_task(task_mode) else BASE_FIELDS


def _body_field(task_mode: str) -> str:
    return BODY_FIELD if is_reference_task(task_mode) else BASE_BODY_FIELD


def _map_masked(text: str, pattern: re.Pattern[str], repl: Callable[[re.Match[str]], str]) -> str:
    """Rewrite through offsets found on the length-preserving literal mask.

    The mask blanks quoted dialogue, lyrics and visible text, so a "(S1)" or
    "[Shot 2]" inside exact source text is never edited. A span that only looks
    like a match because the mask blanked what it contains is skipped: the same
    pattern must also match the real text at those offsets.
    """
    masked = mask_literals(text)
    parts: list[str] = []
    last = 0
    for match in pattern.finditer(masked):
        if not pattern.fullmatch(text, match.start(), match.end()):
            continue
        parts.append(text[last:match.start()])
        parts.append(repl(match))
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def _clean_separators(content: str) -> str:
    """Repair list separators left by a removed reference (used on one clause only)."""
    content = re.sub(r",\s*,", ", ", content)
    content = re.sub(r",[ \t]*$", "", content)
    content = re.sub(r"^[ \t]*,", "", content)
    return re.sub(r"[ \t]{2,}", " ", content).strip()


def _collapse_spaces_outside_literals(text: str) -> str:
    """Collapse spaces left by a removed reference, never inside exact source text."""
    protected = [match.span() for match in LITERAL_RE.finditer(text)]

    def protected_span(start: int, end: int) -> bool:
        return any(left < end and start < right for left, right in protected)

    parts: list[str] = []
    last = 0
    for match in re.finditer(r"[ \t]{2,}", text):
        if protected_span(match.start(), match.end()):
            continue
        parts.append(text[last:match.start()])
        parts.append(" ")
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def _drop_reference(content: str, match: re.Match[str]) -> tuple[int, int]:
    """Extend a deleted reference so its list separator disappears with it."""
    start, end = match.start(), match.end()
    following = re.match(r"[ \t]*,[ \t]*", content[end:])
    if following:
        return start, end + following.end()
    before = re.search(r",\s*$", content[:start])
    if before:
        return before.start(), end
    return start, end


def _rewrite_reference_clause(content: str, mapping: dict[int, int]) -> str | None:
    """Renumber the references of one clause. None means the clause has no references."""
    found = list(SHOT_TOKEN_RE.finditer(content))
    if not found:
        return None
    kept = [item for item in found if int(item.group(1)) in mapping]
    if not kept:
        # The clause exists only to locate a reference in the timeline.
        return ""
    gaps = [content[item.end():nxt.start()] for item, nxt in zip(found, found[1:])]
    if gaps and all(re.fullmatch(r"[ \t]*,[ \t]*", gap) for gap in gaps):
        # A pure reference list is rebuilt from the references this segment owns.
        listed = ", ".join(f"[Shot {mapping[int(item.group(1))]}]" for item in kept)
        return f"{content[:found[0].start()]}{listed}{content[found[-1].end():]}"
    edits: list[tuple[int, int, str]] = []
    for item in found:
        number = int(item.group(1))
        if number in mapping:
            edits.append((item.start(), item.end(), f"[Shot {mapping[number]}]"))
        else:
            start, end = _drop_reference(content, item)
            edits.append((start, end, ""))
    for start, end, replacement in sorted(edits, reverse=True):
        content = content[:start] + replacement + content[end:]
    return _clean_separators(content)


def _rewrite_shot_tokens(text: str, mapping: dict[int, int]) -> str:
    """Map [Shot N] references that belong to this segment and drop the others.

    Clauses and bare references are matched in one scan so a reference is never
    rewritten twice: a reference renamed to this segment's numbering stays renamed.
    A parenthesised shot-placement clause is dropped entirely when none of its
    references belong to this segment, because that clause only exists to locate a
    reference in the timeline.
    """

    def replace(match: re.Match[str]) -> str:
        original = text[match.start():match.end()]
        if match.group(2) is None:
            number = int(match.group(3))
            return f"[Shot {mapping[number]}]" if number in mapping else ""
        clause = REFERENCE_SCAN_RE.fullmatch(original)
        if clause is None:  # defensive: the mask guard already rejects this
            return original
        rewritten = _rewrite_reference_clause(clause.group(2), mapping)
        if rewritten is None:
            return original
        if not rewritten:
            # Dropping the clause also drops the space it hung on.
            return ""
        return f"{clause.group(1)}({rewritten})"

    return _collapse_spaces_outside_literals(_map_masked(text, REFERENCE_SCAN_RE, replace))


def _speaker_map(parts: list[str]) -> dict[str, str]:
    """Number the speakers of a segment in the order they produce a vocal event.

    The segment timeline is scanned first, so whoever speaks first in this segment
    gets (S1). Identifiers that only appear in the analysis sections are added after
    that, in document order; a segment whose first speaker is the source first
    speaker therefore maps every identifier to itself.
    """
    mapping: dict[str, str] = {}
    for part in parts:
        for match in SPEAKER_RE.finditer(mask_literals(part)):
            for raw in match.group(1).split(","):
                key = raw.strip().upper()
                if key not in mapping:
                    mapping[key] = f"S{len(mapping) + 1}"
    return mapping


def _renumber_speakers(text: str, mapping: dict[str, str]) -> str:
    if not mapping:
        return text

    def replace(match: re.Match[str]) -> str:
        members = {mapping.get(raw.strip().upper(), raw.strip().upper()) for raw in match.group(1).split(",")}
        return "(" + ",".join(sorted(members, key=lambda value: int(value[1:]))) + ")"

    return _map_masked(text, SPEAKER_RE, replace)


def _parse_timeline(body: str) -> tuple[str, list[_Shot], bool]:
    """Read the authored timeline.

    `[Shot 1]` may open with no cut time, which is the authored form, or with
    `At 00:00.000,` that a generator sometimes writes for the same instant. The zero
    form is accepted as exactly equivalent to having none: it is consumed here, so a
    segment never inherits it. A first shot that opens at any other time is rejected
    because the part before it belongs to no segment.
    """
    masked = mask_literals(body)
    labels = list(SHOT_RE.finditer(masked))
    if not labels:
        raise _error("正文没有可分段的真实 [Shot 1] 时间线。/ The prompt has no real [Shot 1] timeline to segment.")
    shots: list[_Shot] = []
    opened_at_zero = False
    for position, match in enumerate(labels):
        index = int(match.group(1))
        cut = (FIRST_SHOT_CUT_RE if position == 0 else CUT_RE).match(masked, match.end())
        dropped = False
        if position == 0:
            start_ms, text_start = 0, match.end()
            if cut:
                minutes, seconds, millis = (int(cut.group(part)) for part in (1, 2, 3))
                if minutes or seconds or millis:
                    raise _error(
                        "[Shot 1] 只能没有时间码，或写作等同于开头的 At 00:00.000；请先用质量纠正流程修复原提示词。 "
                        "/ [Shot 1] may carry no cut time or only At 00:00.000, which means the opening instant; "
                        "repair the prompt before segmenting."
                    )
                # The zero opening cut names the same instant as having none, so it is consumed
                # here and no segment inherits it; the flag keeps the space it hung on.
                opened_at_zero, dropped, text_start = True, True, cut.end()
        else:
            if not cut:
                raise _error(
                    f"[Shot {index}] 缺少 At MM:SS.mmm, 切点，无法按时间线分段。 "
                    f"/ [Shot {index}] has no parseable At MM:SS.mmm, cut time."
                )
            minutes, seconds, millis = (int(cut.group(part)) for part in (1, 2, 3))
            if minutes >= 100 or seconds >= 60:
                raise _error(f"[Shot {index}] 的时间码超出 MM:SS.mmm 范围。/ [Shot {index}] uses an unrepresentable cut time.")
            start_ms, text_start = minutes * 60_000 + seconds * 1000 + millis, cut.end()
        end = labels[position + 1].start() if position + 1 < len(labels) else len(body)
        shots.append(_Shot(index, start_ms, text_start, end, dropped))
    starts = [shot.start_ms for shot in shots]
    if any(now <= before for before, now in zip(starts, starts[1:])):
        raise _error("切点时间必须严格递增。/ Cut times must strictly increase.")
    return body[: labels[0].start()], shots, opened_at_zero


def _even_groups(shots: list[_Shot], total_ms: int, count: int) -> list[list[_Shot]]:
    groups: list[list[_Shot]] = [[] for _ in range(count)]
    for shot in shots:
        groups[min(count - 1, (shot.start_ms * count) // total_ms)].append(shot)
    for position, group in enumerate(groups, 1):
        if not group:
            raise _error(
                f"分段 {position} 在 {total_ms / count / 1000:.3f} 秒等分边界内不会包含任何镜头；请求分段数不得超过镜头数 {len(shots)}。 "
                f"/ Segment {position} would contain no shot at {_seconds_text(int(total_ms / count))}s boundaries; "
                f"request at most {len(shots)} segments for this timeline."
            )
    return groups


def _shot_aligned_groups(shots: list[_Shot], total_ms: int, count: int) -> list[list[_Shot]]:
    """Partition the ordered shots into `count` contiguous runs closest to even targets.

    Every boundary lands on a real authored cut time, so no shot is truncated and
    no segment opens on content owned by the previous segment. Ties in total
    deviation are resolved toward the earliest boundary, which keeps the search
    deterministic for a given prompt.
    """
    total = len(shots)
    best: list[list[Fraction | None]] = [[None] * (total + 1) for _ in range(count + 1)]
    parent: list[list[int]] = [[-1] * (total + 1) for _ in range(count + 1)]
    best[0][0] = Fraction(0)
    for group in range(1, count + 1):
        target = Fraction(total_ms * (group - 1), count)
        for covered in range(group, total + 1):
            for start in range(group - 1, covered):
                previous = best[group - 1][start]
                if previous is None:
                    continue
                cost = previous + abs(Fraction(shots[start].start_ms) - target)
                if best[group][covered] is None or cost < best[group][covered]:
                    best[group][covered] = cost
                    parent[group][covered] = start
    if best[count][total] is None:
        raise _error(f"镜头数不足以分成 {count} 段。/ The timeline has too few shots for {count} segments.")
    cursor = total
    cuts = [total]
    for group in range(count, 1, -1):
        cursor = parent[group][cursor]
        cuts.append(cursor)
    cuts.append(0)
    cuts.reverse()
    return [shots[start:end] for start, end in zip(cuts, cuts[1:])]


def _group_bounds(groups: list[list[_Shot]], total_ms: int, split_mode: str) -> list[tuple[int, int]]:
    if split_mode == SPLIT_EVEN:
        count = len(groups)
        return [(int(Fraction(total_ms * position, count)), int(Fraction(total_ms * (position + 1), count)))
                for position in range(count)]
    starts = [group[0].start_ms for group in groups]
    return list(zip(starts, [*starts[1:], total_ms]))


def _alignment_prefix(task_mode: str, group: list[_Shot], owns_first_shot: bool, owns_last_shot: bool,
                     duration_seconds: float, segment_index: int, notes: list[dict[str, Any]]) -> str:
    if task_mode not in KEYFRAME_MODES:
        return ""
    shot_count = len(group)
    if task_mode == "I2VA" and owns_first_shot:
        return alignment_sentence("I2VA", duration_seconds, shot_count)
    if task_mode == "L2VA" and owns_last_shot:
        return alignment_sentence("L2VA", duration_seconds, shot_count)
    if task_mode == "FL2VA" and (owns_first_shot or owns_last_shot):
        clauses = []
        if owns_first_shot:
            clauses.append("Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video")
        if owns_last_shot:
            clauses.append(f"Picture 2 (from Shot {shot_count}) aligns with the {duration_seconds:.2f}-second mark of the target video")
        return "How the reference pictures align with the target video — " + "; ".join(clauses) + "."
    notes.append({
        "code": "segment_without_keyframe_alignment",
        "segment_index": segment_index,
        "severity": "info",
        "message": (
            f"分段 {segment_index} 不承载关键帧角色，因此没有对齐首行，只能作为续接片段使用。 "
            f"/ Segment {segment_index} holds no keyframe role, so it carries no alignment line and can only be used as a continuation segment."
        ),
    })
    return ""


def split_h3_prompt(
    text: str,
    *,
    duration_seconds: Any,
    segment_count: Any,
    task_type: str,
    split_mode: Any = SPLIT_EVEN,
    language: str = "AUTO",
) -> dict[str, Any]:
    """Split one finished H3 prompt into complete per-segment H3 prompts."""
    source = str(text or "")
    task_mode = str(task_type or "")
    if task_mode not in {*TASK_MODES, HYBRID}:
        raise _error(f"Unsupported task_type for segmentation: {task_mode or 'unset'}")
    mode = normalize_split_mode(split_mode)
    try:
        count = int(segment_count)
    except (TypeError, ValueError):
        raise _error("segment_count must be an integer from 1 to 10.") from None
    if not 1 <= count <= MAX_SEGMENTS:
        raise _error(f"segment_count must be an integer from 1 to {MAX_SEGMENTS}.")
    total_ms = _milliseconds(duration_seconds, "duration_seconds")
    if total_ms > int(MAX_TOTAL_SECONDS * 1000):
        raise _error(f"duration_seconds must not exceed {MAX_TOTAL_SECONDS:.0f}s for segmentation.")
    if not source.strip():
        raise _error("Cannot segment an empty prompt.")

    text = repair_shot_timecodes(source)
    parts = sections(text)
    required = _required_fields(task_mode)
    if tuple(part.name for part in parts) != required:
        raise _error(
            "分段需要完整且顺序正确的 H3 字段。/ Segmentation requires the complete H3 field set in the required order: "
            + ", ".join(required)
        )
    body_name = _body_field(task_mode)
    body = next(part.value for part in parts if part.name == body_name)
    head, shots, opened_at_zero = _parse_timeline(body)

    if count == 1:
        groups: list[list[_Shot]] = [shots]
        bounds = [(0, total_ms)]
    else:
        if len(shots) < count:
            raise _error(
                f"时间线只有 {len(shots)} 个镜头，无法分成 {count} 段。 "
                f"/ The timeline has {len(shots)} shots; it cannot be split into {count} segments."
            )
        groups = _even_groups(shots, total_ms, count) if mode == SPLIT_EVEN else _shot_aligned_groups(shots, total_ms, count)
        bounds = _group_bounds(groups, total_ms, mode)

    last_shot_index = shots[-1].index
    body_tail = body[len(body.rstrip()):]
    notes: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    for position, (group, (group_start, group_end)) in enumerate(zip(groups, bounds)):
        segment_index = position + 1
        mapping = {shot.index: number for number, shot in enumerate(group, 1)}
        base_ms = group[0].start_ms
        duration_ms = group_end - group_start
        duration_text = _seconds_text(duration_ms)
        lead_ms = group[0].start_ms - group_start

        if count == 1:
            rendered = {part.name: part.value for part in parts}
            speakers: dict[str, str] = {}
        else:
            values = {part.name: _rewrite_shot_tokens(part.value, mapping) for part in parts}
            rebuilt = [_rewrite_shot_tokens(head, mapping)]
            for number, shot in enumerate(group, 1):
                rebuilt.append(f"[Shot {number}]")
                if number > 1:
                    rebuilt.append(f" At {_clock(shot.start_ms - base_ms)},")
                text = body[shot.text_start:shot.text_end]
                if shot.clause_dropped and text and not text[:1].isspace():
                    # The accepted opening clause was separated from the description by the space
                    # after its comma; keep that separation instead of welding the label to the text.
                    text = " " + text
                rebuilt.append(_rewrite_shot_tokens(text, mapping))
            if group[-1].text_end < len(body):
                # The dropped shots carried the blank line that separated the body
                # from the next section, so restore exactly one blank line.
                rebuilt[-1] = rebuilt[-1].rstrip() + "\n\n"
            else:
                rebuilt[-1] = rebuilt[-1].rstrip() + body_tail
            values[body_name] = "".join(rebuilt)
            speakers = _speaker_map([values[body_name], *[values[part.name] for part in parts if part.name != body_name]])
            rendered = {name: _renumber_speakers(value, speakers) for name, value in values.items()}

        alignment = _alignment_prefix(
            task_mode, group, position == 0, group[-1].index == last_shot_index,
            duration_text, segment_index, notes,
        ) if count > 1 else ""
        prompt = "".join(
            ([f"{alignment}\n\n"] if alignment else []) + [f"{part.name}:{rendered[part.name]}" for part in parts]
        ).strip() if count > 1 else text.strip()

        validation = check_h3(prompt, duration=duration_text, language=language)
        ignored = set(KEYFRAME_ALIGNMENT_CODES)
        if opened_at_zero:
            # The source opened with At 00:00.000, which this module reads as having no cut time
            # at all; _parse_timeline already validated every real cut in the timeline, so the
            # protocol module must not report the accepted opening as a defect of the segment.
            ignored.add("h3_shot_timecode")
        issues = [{"code": item["code"], "message": item["message"], "severity": item["severity"]}
                  for item in validation["issues"] if item["code"] not in ignored]

        if count > 1 and position + 1 < len(groups) and groups[position + 1][0].start_ms > group_end:
            crossing_shot = body[group[-1].text_start:group[-1].text_end]
            notes.append({
                "code": "segment_shot_crosses_boundary",
                "segment_index": segment_index,
                "severity": "warning",
                "contains_exact_text": bool(LITERAL_RE.search(crossing_shot)),
                "message": (
                    f"[Shot {mapping[group[-1].index]}] 原本延续到 {_seconds_text(groups[position + 1][0].start_ms)} 秒，"
                    f"超过本段 {_seconds_text(group_end)} 秒边界：本段会在镜头中途结束，越过边界的画面与台词不会出现在任何一段。 "
                    f"/ Source shot {group[-1].index} runs to {_seconds_text(groups[position + 1][0].start_ms)}s, past this "
                    f"segment's {_seconds_text(group_end)}s boundary: this segment ends mid-shot and the part past the "
                    "boundary appears in no segment."
                ),
            })
        if count > 1 and lead_ms > 0:
            notes.append({
                "code": "segment_lead_without_source",
                "segment_index": segment_index,
                "severity": "warning",
                "message": (
                    f"本段前 {lead_ms / 1000:.3f} 秒在原时间线属于上一段的最后一个镜头，本段文本没有描述它；"
                    "请改用镜头对齐分段，或把本段时长设为其已描述内容的长度。 "
                    f"/ The first {lead_ms / 1000:.3f}s of this segment continues a shot owned by the previous segment; "
                    "use Shot-aligned splitting, or shorten this segment to what it actually describes."
                ),
            })
        if count > 1 and position > 0 and SCENE_TRANS_RE.search(body[group[0].text_start:group[0].text_end]):
            notes.append({
                "code": "segment_opens_with_scenetrans",
                "segment_index": segment_index,
                "severity": "warning",
                "message": (
                    f"本段以 <scenetrans> 延续开始，而它的前半句属于上一段；请人工核对是否需要改为新的开场。 "
                    f"/ This segment opens on a <scenetrans> continuation whose first half belongs to the previous segment."
                ),
            })

        segments.append({
            "segment_index": segment_index,
            "start_seconds": _seconds_text(group_start),
            "end_seconds": _seconds_text(group_end),
            "duration_seconds": duration_text,
            "lead_offset_seconds": _seconds_text(lead_ms),
            "shot_map": [
                {
                    "source_shot": shot.index,
                    "segment_shot": number,
                    "source_start_seconds": _seconds_text(shot.start_ms),
                    "segment_start_seconds": 0.0 if number == 1 else _seconds_text(shot.start_ms - base_ms),
                }
                for number, shot in enumerate(group, 1)
            ],
            "speaker_map": speakers,
            "carries_fields": list(required),
            "deterministic_status": validation["deterministic_status"],
            "validation_issues": issues,
            "prompt": prompt,
        })

    return {
        "schema": SEGMENT_SCHEMA,
        "task_type": task_mode,
        "split_mode": mode,
        "requested_duration_seconds": _seconds_text(total_ms),
        "requested_segment_count": count,
        "source_shot_count": len(shots),
        "segment_count": len(segments),
        "segment_duration_seconds": [_seconds_text(end - start) for start, end in bounds],
        "notes": notes,
        "limits": [
            "分段是对已完成提示词的确定性文本切分：没有重新生成、没有翻译、没有新增事实。",
            "每段都是独立完整的 H3 提示词；跨段的人物身份、道具状态和叙事连续性需要人工或下游连续性工具核对。",
            "平均分段保持请求的等分边界；需要镜头不被切断时改用镜头对齐分段。",
        ],
        "segments": segments,
    }


def segments_json_text(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)


def segments_text(report: dict[str, Any]) -> str:
    """Human-pasteable block; each segment keeps its own complete H3 fields."""
    blocks = [
        "\n".join([
            f"===== T8 SEGMENT {item['segment_index']}/{report['segment_count']} "
            f"| {item['start_seconds']:.3f}-{item['end_seconds']:.3f}s "
            f"| {len(item['shot_map'])} shots =====",
            item["prompt"],
        ])
        for item in report["segments"]
    ]
    return "\n\n".join(blocks)


def _preset_shell() -> dict[str, Any]:
    """The reference preset's fields in the order that file writes them, with the canvas at 768x768.

    Only the values that file derives from its active segment are left for the caller to fill:
    canvas.generation_seconds and timeline.selection. Their numbers there are that segment's
    frame range divided by the preset's fps, and the same arithmetic is applied here.
    """
    return {
        "format": PRESET_FORMAT,
        "schema_version": PRESET_SCHEMA_VERSION,
        "name": PRESET_NAME,
        "preset_type": PRESET_TYPE,
        "canvas": {**PRESET_CANVAS, "generation_seconds": PRESET_SEGMENT_SECONDS, "fps": PRESET_FPS},
        "sampling": dict(PRESET_SAMPLING),
        "timeline": {
            "version": PRESET_TIMELINE_VERSION,
            "fps": PRESET_FPS,
            "globalPrompt": "",
            "secondPass": PRESET_SAMPLING["second_pass"],
            "secondPassHighSteps": PRESET_SAMPLING["high_resolution_steps"],
            "selection": {"start": 0.0, "duration": PRESET_SEGMENT_SECONDS},
            "videoAudioEnabled": True,
            "videoClips": [],
            "images": [],
            "audios": [],
            "segmentConfig": {"count": 0, "activeIndex": 0, "mode": "timeline", "segments": []},
        },
    }


def preset_segment_frames(count: int) -> list[tuple[int, int]]:
    """Pseudo 24 fps ranges for the segments: ten seconds each, each one starting a second inside the previous end."""
    ranges: list[tuple[int, int]] = []
    start = 0
    for _ in range(max(0, int(count))):
        ranges.append((start, start + PRESET_SEGMENT_FRAMES))
        start += PRESET_SEGMENT_FRAMES - PRESET_OVERLAP_FRAMES
    return ranges


def segments_preset_json(report: dict[str, Any]) -> str:
    """The segment prompts as a minimax-h3-timeline-preset, or an empty string when there are none.

    Every segment entry carries only that segment's own generated prompt; the image and audio
    lists stay empty because segmentation has no media to name. The frame numbers are pseudo
    timeline positions in the preset's 24 fps, not measurements of the finished video.
    """
    items = [item for item in (report.get("segments") or []) if str(item.get("prompt") or "")]
    ranges = preset_segment_frames(len(items))
    if not items or not ranges:
        return ""
    preset = _preset_shell()
    active = 0
    active_start, active_end = ranges[active]
    active_seconds = (active_end - active_start) / PRESET_FPS
    preset["canvas"]["generation_seconds"] = active_seconds
    timeline = preset["timeline"]
    timeline["selection"] = {"start": active_start / PRESET_FPS, "duration": active_seconds}
    config = timeline["segmentConfig"]
    config["count"] = len(items)
    config["activeIndex"] = active
    config["segments"] = [
        {"images": [], "audios": [], "prompt": item["prompt"], "startFrame": frame_start, "endFrame": frame_end}
        for item, (frame_start, frame_end) in zip(items, ranges)
    ]
    return json.dumps(preset, ensure_ascii=False, indent=2)


__all__ = [
    "MAX_SEGMENTS",
    "PRESET_FORMAT",
    "PRESET_FPS",
    "PRESET_OVERLAP_FRAMES",
    "PRESET_SCHEMA_VERSION",
    "PRESET_SEGMENT_FRAMES",
    "PRESET_SEGMENT_SECONDS",
    "SegmentSplitError",
    "SEGMENT_SCHEMA",
    "SPLIT_EVEN",
    "SPLIT_MODE_ALIASES",
    "SPLIT_MODE_OPTIONS",
    "SPLIT_SHOT_ALIGNED",
    "normalize_split_mode",
    "preset_segment_frames",
    "segments_json_text",
    "segments_preset_json",
    "segments_text",
    "split_h3_prompt",
]
