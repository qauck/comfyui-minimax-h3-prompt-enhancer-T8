"""Deterministic timeline segmentation of a finished H3 prompt (pure CPU, no provider).

The reference case is the documented request: a 20-second Ref2VA timeline with four
shots at 0/5/11/16s split into two segments. Segment 1 owns shots 1-2 and segment 2
owns shots 3-4, renumbered so each segment starts at [Shot 1] with no cut time and
keeps the exact intervals it had in the source timeline.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_segments import (  # noqa: E402
    MAX_SEGMENTS,
    PRESET_FPS,
    PRESET_OVERLAP_FRAMES,
    PRESET_SEGMENT_FRAMES,
    PRESET_SEGMENT_SECONDS,
    SEGMENT_SCHEMA,
    SPLIT_EVEN,
    SPLIT_SHOT_ALIGNED,
    SegmentSplitError,
    normalize_split_mode,
    preset_segment_frames,
    segments_json_text,
    segments_preset_json,
    segments_text,
    split_h3_prompt,
)
from h3_quality import LITERAL_RE, check_h3, repair_shot_timecodes, sections  # noqa: E402

REFERENCE_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

# 20s Ref2VA timeline, four shots at 00:00 / 00:05 / 00:11 / 00:16.
REF2VA_20S = """subject_definitions:
<Subject 1> is the analyst in the grey coat whose identity and wardrobe come from <Picture 1>.
<Subject 2> is the delivery drone whose form comes from <Video 1>.
<Subject 3> is the counter clerk (S2) whose voice quality is referenced from <Video 1>.
<Subject 4> is the supervisor (S1) who speaks off-screen in the opening.
<Picture 1> is the identity and composition reference for <Subject 1>.
<Video 1> supplies the approach path and the cut rhythm.

summary:
[reference generation] A twenty-second station scene in which <Subject 4> (S1) briefs the analyst while the drone crosses the platform.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3], [Shot 4]): fully_preserved - identity, wardrobe and proportions are retained throughout.
<Subject 2> (appears in [Shot 3]): weak_reference - only the approach path from <Video 1> guides the target motion.
<Subject 3> (appears in [Shot 3], [Shot 4]): weak_reference - only the voice quality referenced from <Video 1> guides the clerk.
<Subject 4> (appears in [Shot 1], [Shot 2]): attribute_transfer - only the off-screen voice quality transfers to an original line.
<Picture 1> ([Shot 1] identity reference): attribute_transfer - identity and wardrobe transfer while the station becomes an original scene.
<Video 1> (approach and cut structure): weak_reference - its temporal order guides the target sequence.

detailed_description:
Natural documentary image quality with soft daylight and restrained handheld motion.
[Shot 1] <Subject 4> (S1) says in an off-screen voiceover, <d>[English] Keep the segment order exact.</d> while <Subject 1> walks along the platform reading the sign "STATION-9".
[Shot 2] At 00:05.000, the camera side-walks to a medium two-shot as <Subject 1> (S1) turns toward the kiosk.
[Shot 3] At 00:11.000, <Subject 2> crosses the platform left to right behind the counter. <Subject 3> (S2) says in a flat middle-aged voice, <d>[English] The next window opens at nine.</d>
[Shot 4] At 00:16.000, <Subject 4> (S1) replies off-screen, <d>[English] Noted.</d> <Subject 3> (S2) nods once and the frame holds.

overall_soundscape:
Continuous station room tone with distant announcements, footsteps and a soft pneumatic door hiss.

non_diegetic_music:
A sparse piano figure at slow tempo with light string pads stays below the dialogue."""

BASE_12S = """integrated_multimodal_description:
[Shot 1] A courier in a green jacket lifts a crate in a loading bay. The worker (S1) says, <d>[English] Load the last one.</d>
[Shot 2] At 00:04.000, the crate slides onto the belt and the worker (S1) says, <d>[English] Send it.</d>
[Shot 3] At 00:08.000, the belt carries the crate out of frame and the bay lights dim.

overall_soundscape:
Loading bay room tone with belt motors and distant forklift beeps.

non_diegetic_music:
A low pulsing synth bed stays under the scene."""

I2VA_16S = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description:
[Shot 1] <Picture 1> is fully referenced as the opening frame; a woman in a red coat stands at a window.
[Shot 2] At 00:06.000, she turns toward the door.
[Shot 3] At 00:12.000, she opens the door and the hallway light spills in.

overall_soundscape:
Quiet room tone with cloth movement and a latch click.

non_diegetic_music:
No non-diegetic music."""


def field_names(text: str) -> tuple[str, ...]:
    return tuple(part.name for part in sections(text))


def body_of(text: str) -> str:
    return next(part.value for part in sections(text) if part.name in {"detailed_description", "integrated_multimodal_description"})


class SplitReferenceCase(unittest.TestCase):
    """The requested behaviour: two complete prompts from one 20s four-shot prompt."""

    def report(self, count: int = 2, mode: str = SPLIT_EVEN) -> dict:
        return split_h3_prompt(REF2VA_20S, duration_seconds=20, segment_count=count,
                              task_type="Ref2VA", split_mode=mode, language="English")

    def test_two_segments_follow_the_timeline_and_rebase_intervals(self):
        report = self.report()
        self.assertEqual(report["schema"], SEGMENT_SCHEMA)
        self.assertEqual(report["segment_count"], 2)
        self.assertEqual(report["source_shot_count"], 4)
        self.assertEqual([item["duration_seconds"] for item in report["segments"]], [10.0, 10.0])
        self.assertEqual([(item["start_seconds"], item["end_seconds"]) for item in report["segments"]],
                         [(0.0, 10.0), (10.0, 20.0)])

        first, second = report["segments"]
        self.assertEqual([(item["source_shot"], item["segment_shot"], item["segment_start_seconds"])
                          for item in first["shot_map"]], [(1, 1, 0.0), (2, 2, 5.0)])
        # The user's own calculation: 20 / 2 * 1 = 10s offset, so the shot that
        # started at 11s opens segment 2 at 0s and the 16s shot lands 5s later.
        self.assertEqual([(item["source_shot"], item["segment_shot"], item["source_start_seconds"],
                          item["segment_start_seconds"]) for item in second["shot_map"]],
                         [(3, 1, 11.0, 0.0), (4, 2, 16.0, 5.0)])
        self.assertEqual(second["lead_offset_seconds"], 1.0)
        self.assertEqual(first["lead_offset_seconds"], 0.0)

    def test_each_segment_is_a_complete_prompt_with_its_own_numbering(self):
        report = self.report()
        for item in report["segments"]:
            prompt = item["prompt"]
            self.assertEqual(field_names(prompt), REFERENCE_FIELDS)
            self.assertTrue(prompt.startswith("subject_definitions:"))
            self.assertEqual(item["carries_fields"], list(REFERENCE_FIELDS))
            self.assertIn("[Shot 1] ", body_of(prompt))
            self.assertIn("[Shot 2] At 00:05.000,", body_of(prompt))
            self.assertNotIn("[Shot 3]", prompt)
            self.assertNotIn("[Shot 4]", prompt)
            self.assertNotIn("At 00:11.000", prompt)
            self.assertNotIn("At 00:16.000", prompt)
            # The cut time of [Shot 1] never appears: the timeline starts at zero.
            self.assertNotIn("[Shot 1] At", body_of(prompt))
            self.assertEqual(item["deterministic_status"], "passed_checked_scope")
            self.assertEqual([issue["code"] for issue in item["validation_issues"]], [])

    def test_shot_and_speaker_references_are_renumbered_per_segment(self):
        first, second = self.report()["segments"]
        # Each segment carries the same analysis text, but every [Shot N] reference is
        # rewritten to that segment's own numbering, so a four-shot appearance list
        # becomes the two shots this segment actually contains.
        self.assertIn("<Subject 1> (appears in [Shot 1], [Shot 2])", first["prompt"])
        self.assertIn("<Picture 1> ([Shot 1] identity reference)", first["prompt"])
        # Segment 2 renumbers the list to its own two shots and drops the clause that
        # only existed to place Picture 1 at the source first shot.
        self.assertIn("<Subject 1> (appears in [Shot 1], [Shot 2])", second["prompt"])
        self.assertIn("<Subject 2> (appears in [Shot 1])", second["prompt"])
        self.assertIn("<Picture 1>: attribute_transfer", second["prompt"])
        self.assertNotIn("([Shot 1] identity reference)", second["prompt"])
        # A placement clause whose references all live in the other segment is dropped,
        # together with the space it hung on, so the retention line still reads correctly.
        self.assertIn("<Subject 3>: weak_reference", first["prompt"])
        self.assertIn("<Subject 4>: attribute_transfer", second["prompt"])
        self.assertIn("<Subject 2>: weak_reference", first["prompt"])
        # In segment 2 the clerk speaks first, so (S2) becomes (S1) for that segment.
        # Segment 1 still opens with the supervisor, and the clerk is only named in the
        # analysis sections there, so both identifiers keep the numbering they carry.
        self.assertEqual(first["speaker_map"], {"S1": "S1", "S2": "S2"})
        self.assertEqual(second["speaker_map"], {"S2": "S1", "S1": "S2"})
        self.assertIn("<Subject 3> (S1) says in a flat middle-aged voice", second["prompt"])
        self.assertIn("<Subject 4> (S2) replies off-screen", second["prompt"])

    def test_exact_text_survives_byte_for_byte(self):
        first, second = self.report()["segments"]
        self.assertIn('<d>[English] Keep the segment order exact.</d>', first["prompt"])
        self.assertIn('<d>[English] The next window opens at nine.</d>', second["prompt"])
        self.assertIn('<d>[English] Noted.</d>', second["prompt"])
        self.assertIn('"STATION-9"', first["prompt"])
        self.assertNotIn("The next window opens at nine", first["prompt"])
        # Renumbering never edits a speaker token that sits inside exact text.
        self.assertNotIn("(S2) says in a flat middle-aged voice", second["prompt"].split("<d>")[1][:40])

    def test_even_split_reports_the_boundary_costs_it_cannot_hide(self):
        report = self.report()
        codes = {(note["code"], note.get("segment_index")) for note in report["notes"]}
        self.assertIn(("segment_shot_crosses_boundary", 1), codes)
        self.assertIn(("segment_lead_without_source", 2), codes)
        self.assertIn("分段", " ".join(note["message"] for note in report["notes"]))

    def test_shot_aligned_split_never_truncates_a_shot(self):
        report = self.report(2, SPLIT_SHOT_ALIGNED)
        first, second = report["segments"]
        self.assertEqual([(first["start_seconds"], first["end_seconds"]), (second["start_seconds"], second["end_seconds"])],
                         [(0.0, 11.0), (11.0, 20.0)])
        self.assertEqual([item["duration_seconds"] for item in report["segments"]], [11.0, 9.0])
        self.assertEqual([item["lead_offset_seconds"] for item in report["segments"]], [0.0, 0.0])
        codes = {note["code"] for note in report["notes"]}
        self.assertNotIn("segment_shot_crosses_boundary", codes)
        self.assertNotIn("segment_lead_without_source", codes)
        # The same rebase rule applies: the final shot is 5s after the opening shot.
        self.assertIn("[Shot 2] At 00:05.000,", body_of(second["prompt"]))
        self.assertEqual(report["segment_duration_seconds"], [11.0, 9.0])

    def test_more_segments_than_shots_fails_closed(self):
        for count in (5, 6):
            with self.subTest(count=count):
                with self.assertRaises(SegmentSplitError) as raised:
                    self.report(count)
                self.assertIn("镜头", str(raised.exception))

    def test_single_segment_is_the_original_prompt(self):
        report = self.report(1)
        self.assertEqual(report["segment_count"], 1)
        self.assertEqual(report["segments"][0]["prompt"], REF2VA_20S)
        self.assertEqual(report["segments"][0]["speaker_map"], {})
        self.assertEqual(report["notes"], [])

    def test_outputs_are_json_and_pasteable_text(self):
        report = self.report()
        decoded = json.loads(segments_json_text(report))
        self.assertEqual(decoded["segment_count"], 2)
        text = segments_text(report)
        self.assertIn("===== T8 SEGMENT 1/2 | 0.000-10.000s | 2 shots =====", text)
        self.assertIn("===== T8 SEGMENT 2/2 | 10.000-20.000s | 2 shots =====", text)
        self.assertEqual(text.count("subject_definitions:"), 2)

    def test_fullwidth_cut_comma_is_repaired_before_grouping(self):
        broken = REF2VA_20S.replace("At 00:11.000,", "At 00:11.000，").replace("At 00:16.000,", "At 00:16.000，")
        report = split_h3_prompt(broken, duration_seconds=20, segment_count=2, task_type="Ref2VA", language="English")
        self.assertEqual([item["duration_seconds"] for item in report["segments"]], [10.0, 10.0])
        self.assertIn("[Shot 2] At 00:05.000,", body_of(report["segments"][1]["prompt"]))

    def test_fractional_duration_splits_on_exact_millisecond_boundaries(self):
        report = split_h3_prompt(REF2VA_20S, duration_seconds=20 / 3, segment_count=2, task_type="Ref2VA", language="English")
        self.assertEqual(report["requested_duration_seconds"], 6.667)
        self.assertEqual([item["duration_seconds"] for item in report["segments"]], [3.333, 3.334])
        # Shots at 5, 11 and 16s exceed a 6.667s timeline, so grouping fails closed.
        with self.assertRaises(SegmentSplitError):
            split_h3_prompt(REF2VA_20S, duration_seconds=20, segment_count=MAX_SEGMENTS + 1, task_type="Ref2VA")


class SplitBaseModes(unittest.TestCase):
    def test_three_field_prompt_splits_with_the_same_rules(self):
        report = split_h3_prompt(BASE_12S, duration_seconds=12, segment_count=2, task_type="T2VA", language="English")
        first, second = report["segments"]
        self.assertEqual(field_names(first["prompt"]), ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"))
        self.assertEqual([(item["source_shot"], item["segment_shot"]) for item in second["shot_map"]], [(3, 1)])
        self.assertIn("the bay lights dim", second["prompt"])
        self.assertNotIn("[Shot 2]", body_of(second["prompt"]))
        self.assertEqual(second["deterministic_status"], "passed_checked_scope")
        self.assertEqual([issue["code"] for issue in second["validation_issues"]], [])

    def test_keyframe_alignment_follows_the_segment_that_owns_the_frame(self):
        report = split_h3_prompt(I2VA_16S, duration_seconds=16, segment_count=2, task_type="I2VA", split_mode=SPLIT_SHOT_ALIGNED, language="English")
        first, second = report["segments"]
        self.assertEqual(first["prompt"].splitlines()[0],
                         "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.")
        self.assertTrue(first["prompt"].startswith("For the target video,"))
        # Segment 2 owns no keyframe, so it carries no alignment line and says so.
        self.assertFalse(second["prompt"].startswith("For the target video,"))
        self.assertIn("integrated_multimodal_description:", second["prompt"])
        self.assertEqual({note["code"] for note in report["notes"]}, {"segment_without_keyframe_alignment"})
        self.assertEqual([issue["code"] for issue in second["validation_issues"]], [])

    def test_split_rejects_incomplete_or_mismatched_field_sets(self):
        with self.assertRaises(SegmentSplitError) as raised:
            split_h3_prompt(BASE_12S, duration_seconds=12, segment_count=2, task_type="Ref2VA")
        self.assertIn("subject_definitions", str(raised.exception))
        with self.assertRaises(SegmentSplitError):
            split_h3_prompt(REF2VA_20S, duration_seconds=20, segment_count=2, task_type="T2VA")

    def test_split_rejects_bad_parameters(self):
        for kwargs in (
            {"duration_seconds": 0, "segment_count": 2, "task_type": "T2VA"},
            {"duration_seconds": -1, "segment_count": 2, "task_type": "T2VA"},
            {"duration_seconds": 12, "segment_count": 0, "task_type": "T2VA"},
            {"duration_seconds": 12, "segment_count": 11, "task_type": "T2VA"},
            {"duration_seconds": 12, "segment_count": 2, "task_type": "Nope"},
            {"duration_seconds": 12, "segment_count": 2, "task_type": "T2VA", "split_mode": "whatever"},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(SegmentSplitError):
                    split_h3_prompt(BASE_12S, **kwargs)
        with self.assertRaises(SegmentSplitError):
            split_h3_prompt("", duration_seconds=12, segment_count=2, task_type="T2VA")

    def test_a_shot_without_a_cut_time_cannot_be_grouped(self):
        broken = BASE_12S.replace("[Shot 3] At 00:08.000,", "[Shot 3]")
        with self.assertRaises(SegmentSplitError) as raised:
            split_h3_prompt(broken, duration_seconds=12, segment_count=2, task_type="T2VA")
        self.assertIn("Shot 3", str(raised.exception))

    def test_a_zero_opening_cut_is_the_same_timeline_as_having_none(self):
        plain = split_h3_prompt(BASE_12S, duration_seconds=12, segment_count=2, task_type="T2VA", language="English")
        for marker in ("[Shot 1] At 00:00.000, A courier", "[Shot 1] At 00:00.000，A courier"):
            opening = BASE_12S.replace("[Shot 1] A courier", marker)
            with self.subTest(opening=marker):
                marked = split_h3_prompt(opening, duration_seconds=12, segment_count=2, task_type="T2VA", language="English")
                self.assertEqual([item["prompt"] for item in marked["segments"]], [item["prompt"] for item in plain["segments"]])
                self.assertEqual([item["shot_map"] for item in marked["segments"]], [item["shot_map"] for item in plain["segments"]])
                self.assertEqual(marked["notes"], plain["notes"])
                for item in marked["segments"]:
                    self.assertNotIn("[Shot 1] At 00:00.000", item["prompt"])
                    self.assertEqual([issue["code"] for issue in item["validation_issues"]], [])
                single = split_h3_prompt(opening, duration_seconds=12, segment_count=1, task_type="T2VA", language="English")
                # A segment that is the whole prompt keeps the author's text, and the accepted
                # opening cut is not reported back as a defect of that segment.
                self.assertEqual(single["segments"][0]["prompt"], repair_shot_timecodes(opening).strip())
                self.assertIn("[Shot 1] At 00:00.000", single["segments"][0]["prompt"])
                self.assertEqual([issue["code"] for issue in single["segments"][0]["validation_issues"]], [])

    def test_a_first_shot_that_opens_after_zero_still_cannot_be_segmented(self):
        for marker in ("[Shot 1] At 00:00.500, A courier", "[Shot 1] At 00:01.000, A courier", "[Shot 1] At 00:00.001，A courier"):
            with self.subTest(opening=marker):
                with self.assertRaises(SegmentSplitError) as raised:
                    split_h3_prompt(BASE_12S.replace("[Shot 1] A courier", marker),
                                    duration_seconds=12, segment_count=2, task_type="T2VA")
                self.assertIn("[Shot 1]", str(raised.exception))
                self.assertIn("00:00.000", str(raised.exception))

    def test_split_mode_aliases(self):
        self.assertEqual(normalize_split_mode(), SPLIT_EVEN)
        self.assertEqual(normalize_split_mode("平均分段（按分段数） / Even"), SPLIT_EVEN)
        self.assertEqual(normalize_split_mode("shot-aligned"), SPLIT_SHOT_ALIGNED)
        with self.assertRaises(SegmentSplitError):
            normalize_split_mode("random")


class SplitSafety(unittest.TestCase):
    def test_module_is_cpu_only(self):
        source = (ROOT / "h3_segments.py").read_text(encoding="utf-8")
        for forbidden in ("requests", "comfy", "torch", "provider", "httpx", "socket", "minimax"):
            self.assertNotIn(f"import {forbidden}", source.lower())
        # The product name may appear only as data copied from the preset document it mirrors:
        # that format identifier, that document's own name, and the path it was read from.
        # Nothing in this module may reach the provider, a network client or ComfyUI.
        copied_from_the_preset = ("minimax-h3-timeline-preset", "minimaxh3_presets", "minimaxh3_preset")
        for line in source.lower().splitlines():
            if "minimax" in line:
                self.assertTrue(any(item in line for item in copied_from_the_preset), line)

    def test_a_shot_the_boundary_cuts_in_half_is_reported(self):
        # Shots at 0/4/8s in a 12s timeline: the even boundary at 6.00s falls inside
        # shot 2, which carries a line of exact dialogue.
        report = split_h3_prompt(BASE_12S, duration_seconds=12, segment_count=2, task_type="T2VA", language="English")
        self.assertEqual([(item["start_seconds"], item["end_seconds"]) for item in report["segments"]], [(0.0, 6.0), (6.0, 12.0)])
        crossing = [note for note in report["notes"] if note["code"] == "segment_shot_crosses_boundary"]
        self.assertEqual([note["segment_index"] for note in crossing], [1])
        self.assertTrue(crossing[0]["contains_exact_text"])
        self.assertIn("segment_lead_without_source", {note["code"] for note in report["notes"]})
        # The same timeline under Shot-aligned splitting keeps every shot whole.
        aligned = split_h3_prompt(BASE_12S, duration_seconds=12, segment_count=2, task_type="T2VA",
                                 split_mode=SPLIT_SHOT_ALIGNED, language="English")
        self.assertEqual({note["code"] for note in aligned["notes"]}, set())
        self.assertEqual([(item["start_seconds"], item["end_seconds"]) for item in aligned["segments"]], [(0.0, 4.0), (4.0, 12.0)])

    def test_rewriting_never_edits_exact_source_text(self):
        source_literals = [match.group() for match in LITERAL_RE.finditer(REF2VA_20S)]
        report = split_h3_prompt(REF2VA_20S, duration_seconds=20, segment_count=3, task_type="Ref2VA",
                                 split_mode=SPLIT_SHOT_ALIGNED, language="English")
        seen: list[str] = []
        for item in report["segments"]:
            literals = [match.group() for match in LITERAL_RE.finditer(item["prompt"])]
            # Every literal in a segment is verbatim source text, in source order.
            cursor = iter(source_literals)
            self.assertTrue(all(words in cursor for words in literals), item["prompt"])
            seen.extend(literals)
        # Splitting distributes the exact text; it never drops or invents any of it.
        self.assertEqual(sorted(seen), sorted(source_literals))

    def test_segments_are_validated_against_protocol_only_not_against_an_absent_source(self):
        report = split_h3_prompt(REF2VA_20S, duration_seconds=20, segment_count=2, task_type="Ref2VA", language="English")
        for item in report["segments"]:
            checked = check_h3(item["prompt"], duration=item["duration_seconds"], language="English")
            self.assertFalse(any(code.startswith("semantic_") for code in {issue["code"] for issue in checked["issues"]}))

    def test_scenetrans_opening_is_flagged_for_the_continuation_segment(self):
        crossing = REF2VA_20S.replace(
            "<Subject 3> (S2) says in a flat middle-aged voice, <d>[English] The next window opens at nine.</d>",
            "<Subject 3> (S2) continues the line started before the cut, <d>[English] <scenetrans>The next window opens at nine.</d>",
        )
        report = split_h3_prompt(crossing, duration_seconds=20, segment_count=2, task_type="Ref2VA", language="English")
        flagged = [note for note in report["notes"] if note["code"] == "segment_opens_with_scenetrans"]
        self.assertEqual([note["segment_index"] for note in flagged], [2])


class SplitPresetOutput(unittest.TestCase):
    """The same segments as a minimax-h3-timeline-preset, in the shape of the reference preset document."""

    def report(self):
        return split_h3_prompt(REF2VA_20S, duration_seconds=20, segment_count=2, task_type="Ref2VA", language="English")

    def test_frame_ranges_are_ten_seconds_each_and_overlap_the_previous_segment(self):
        self.assertEqual(PRESET_SEGMENT_FRAMES, int(PRESET_SEGMENT_SECONDS * PRESET_FPS))
        self.assertEqual(preset_segment_frames(3), [(0, 240), (216, 456), (432, 672)])
        ranges = preset_segment_frames(MAX_SEGMENTS)
        self.assertEqual(len(ranges), MAX_SEGMENTS)
        for start, end in ranges:
            self.assertGreater(end, start)
            self.assertEqual(end - start, PRESET_SEGMENT_FRAMES)
        self.assertEqual([item[0] for item in ranges[1:]],
                         [item[1] - PRESET_OVERLAP_FRAMES for item in ranges[:-1]])
        self.assertEqual(preset_segment_frames(0), [])

    def test_a_report_without_segments_produces_no_preset(self):
        self.assertEqual(segments_preset_json({"segments": []}), "")
        self.assertEqual(segments_preset_json({"schema": SEGMENT_SCHEMA, "status": "not_split", "segment_count": 0,
                                               "error": "no split"}), "")

    def test_the_preset_uses_the_reference_document_fields_with_a_768_canvas(self):
        preset = json.loads(segments_preset_json(self.report()))
        self.assertEqual(list(preset), ["format", "schema_version", "name", "preset_type", "canvas", "sampling", "timeline"])
        self.assertEqual([preset["format"], preset["schema_version"], preset["name"], preset["preset_type"]],
                         ["minimax-h3-timeline-preset", 1, "MiniMaxH3_Preset", "complete"])
        # The canvas is this node's own 768x768 default; the reference document's other fields stand.
        self.assertEqual(preset["canvas"], {"width": 768, "height": 768, "generation_seconds": 10.0, "fps": 24})
        self.assertEqual(preset["sampling"], {"second_pass": True, "high_resolution_steps": 2})
        self.assertEqual(list(preset["timeline"]),
                         ["version", "fps", "globalPrompt", "secondPass", "secondPassHighSteps", "selection",
                          "videoAudioEnabled", "videoClips", "images", "audios", "segmentConfig"])
        timeline = preset["timeline"]
        self.assertEqual([timeline["version"], timeline["fps"], timeline["globalPrompt"]], [10, 24, ""])
        self.assertEqual([timeline["secondPass"], timeline["secondPassHighSteps"], timeline["videoAudioEnabled"]],
                         [True, 2, True])
        self.assertEqual([timeline["videoClips"], timeline["images"], timeline["audios"]], [[], [], []])
        # The reference document states its selection in seconds, matching the active segment's own frame range.
        self.assertEqual(timeline["selection"], {"start": 0.0, "duration": 10.0})
        config = timeline["segmentConfig"]
        self.assertEqual(list(config), ["count", "activeIndex", "mode", "segments"])
        self.assertEqual([config["count"], config["activeIndex"], config["mode"]], [2, 0, "timeline"])

    def test_each_preset_segment_carries_the_generated_prompt_and_no_media(self):
        report = self.report()
        config = json.loads(segments_preset_json(report))["timeline"]["segmentConfig"]
        for entry, item, frame_range in zip(config["segments"], report["segments"], preset_segment_frames(2)):
            # The reference document orders each segment entry exactly like this.
            self.assertEqual(list(entry), ["images", "audios", "prompt", "startFrame", "endFrame"])
            self.assertEqual(entry["images"], [])
            self.assertEqual(entry["audios"], [])
            self.assertEqual(entry["prompt"], item["prompt"])
            self.assertEqual([entry["startFrame"], entry["endFrame"]], list(frame_range))
            self.assertIn("subject_definitions:", entry["prompt"])

    def test_the_frame_ranges_follow_the_segment_count_not_the_authored_timeline(self):
        for count in (3, 4):
            report = split_h3_prompt(REF2VA_20S, duration_seconds=20, segment_count=count, task_type="Ref2VA",
                                     split_mode=SPLIT_SHOT_ALIGNED, language="English")
            config = json.loads(segments_preset_json(report))["timeline"]["segmentConfig"]
            with self.subTest(count=count):
                self.assertEqual(config["count"], count)
                self.assertEqual([(item["startFrame"], item["endFrame"]) for item in config["segments"]],
                                 preset_segment_frames(count))


class NodeSegmentScope(unittest.TestCase):
    """Segmentation belongs to the H3 enhancer only, and segment_count 1 changes nothing."""

    @classmethod
    def setUpClass(cls):
        import asyncio
        import importlib.util

        sys.path.insert(0, str(ROOT.parents[1]))
        spec = importlib.util.spec_from_file_location(
            "t8_h3_segment_scope_package", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        cls.nodes = sys.modules[f"{spec.name}.nodes"]
        extension = asyncio.run(package.comfy_entrypoint())
        cls.schemas = [node.define_schema() for node in asyncio.run(extension.get_node_list())]

    def h3_schema(self):
        return next(schema for schema in self.schemas if schema.node_id == "MiniMaxH3PromptEnhancerT8")

    def inputs(self, **extra):
        return dict(prompt="A woman returns a ticket.", task_type="T2VA", duration_seconds=8,
                    rewrite_mode="balanced", description_word_target=0, output_language="English", **extra)

    def test_only_the_h3_enhancer_exposes_segment_controls(self):
        owners = [schema.node_id for schema in self.schemas
                  if {"segment_count", "segment_split_mode"}.intersection(item.id for item in schema.inputs)]
        self.assertEqual(owners, ["MiniMaxH3PromptEnhancerT8"])
        segment_ports = [schema.node_id for schema in self.schemas
                         if {"h3_segments_text"}.intersection(item.display_name for item in schema.outputs)]
        self.assertEqual(segment_ports, ["MiniMaxH3PromptEnhancerT8"])
        # The long-form planner already published an h3_segments_json port; segmentation did not move into it.
        planner = next(schema for schema in self.schemas if schema.node_id == "T8LongFormPlanner")
        self.assertFalse({"segment_count", "segment_split_mode"}.intersection(item.id for item in planner.inputs))

    def test_segment_controls_follow_every_widget_and_precede_the_opt_in_sockets(self):
        schema = self.h3_schema()
        ids = [item.id for item in schema.inputs]
        offset = ids.index("combat_camera_config")
        # The two controls follow the last published widget position; everything appended after
        # them is an opt-in lazy connection, so the serialized widget array still ends with them.
        self.assertEqual(ids[offset + 1:offset + 3], ["segment_count", "segment_split_mode"])
        self.assertTrue(all(getattr(item, "lazy", False) for item in schema.inputs[offset + 3:]))
        self.assertEqual([item.id for item in schema.inputs if not getattr(item, "lazy", False)][-2:],
                         ["segment_count", "segment_split_mode"])
        self.assertEqual([output.display_name for output in schema.outputs][:6],
                         ["enhanced_prompt", "global_prompt", "local_prompts", "time_ranges", "relay_length", "relay_report"])
        self.assertEqual([output.display_name for output in schema.outputs][6:],
                         ["h3_segments_json", "h3_segments_text", "h3_segments_preset"])

    def test_segment_widgets_default_to_no_splitting(self):
        inputs = {item.id: item for item in self.h3_schema().inputs}
        count = inputs["segment_count"]
        mode = inputs["segment_split_mode"]
        self.assertEqual((count.default, count.min, count.max), (1, 1, MAX_SEGMENTS))
        self.assertEqual((mode.default, mode.options[0]), (SPLIT_EVEN, SPLIT_EVEN))
        self.assertTrue(count.optional and mode.optional)

    def test_segment_count_one_keeps_the_untouched_node_result(self):
        draft = ("integrated_multimodal_description: [Shot 1] A woman returns a ticket.\n"
                 "overall_soundscape: Station ambience.\nnon_diegetic_music: N/A")
        legacy = (draft, "", "", "", 0, "")
        for overrides in ({}, {"segment_count": 1}, {"segment_count": "1"},
                          {"segment_count": 1, "segment_split_mode": SPLIT_SHOT_ALIGNED}):
            with self.subTest(overrides=overrides or "widget defaults"):
                with patch.object(self.nodes, "enhance_prompt", return_value=draft) as completion, \
                        patch.object(self.nodes, "split_h3_prompt",
                                     side_effect=AssertionError("segment_count 1 must not segment the prompt")) as splitter:
                    result = self.nodes.MiniMaxH3PromptEnhancer.execute(**self.inputs(**overrides))
                self.assertEqual(tuple(result.result)[:6], legacy)
                self.assertEqual(tuple(result.result)[6:], ("", "", ""))
                completion.assert_called_once()
                splitter.assert_not_called()

    def test_a_prompt_that_opens_with_a_zero_cut_segments_through_the_node(self):
        draft = ("integrated_multimodal_description: [Shot 1] At 00:00.000, A woman returns a ticket at the counter.\n"
                 "[Shot 2] At 00:04.000, She walks out through the turnstile.\n\n"
                 "overall_soundscape: Station ambience.\nnon_diegetic_music: N/A")
        with patch.object(self.nodes, "enhance_prompt", return_value=draft):
            result = self.nodes.MiniMaxH3PromptEnhancer.execute(**self.inputs(segment_count=2))
        self.assertEqual(tuple(result.result)[:6], (draft, "", "", "", 0, ""))
        report = json.loads(result.result[6])
        self.assertEqual(report["segment_count"], 2)
        self.assertEqual(report["segment_duration_seconds"], [4.0, 4.0])
        first, second = report["segments"]
        self.assertEqual([(entry["source_shot"], entry["segment_shot"], entry["segment_start_seconds"])
                          for entry in first["shot_map"]], [(1, 1, 0.0)])
        self.assertEqual([(entry["source_shot"], entry["segment_shot"], entry["segment_start_seconds"])
                          for entry in second["shot_map"]], [(2, 1, 0.0)])
        self.assertEqual((first["start_seconds"], first["end_seconds"]), (0.0, 4.0))
        for segment in (first, second):
            self.assertNotIn("At 00:00.000", segment["prompt"])
            self.assertEqual([issue["code"] for issue in segment["validation_issues"]], [])
        self.assertIn("[Shot 1] A woman returns a ticket at the counter.", first["prompt"])
        self.assertIn("[Shot 1] She walks out through the turnstile.", second["prompt"])
        self.assertIn("===== T8 SEGMENT 2/2 | 4.000-8.000s | 1 shots =====", result.result[7])
        preset = json.loads(result.result[8])
        self.assertEqual([item["prompt"] for item in preset["timeline"]["segmentConfig"]["segments"]],
                         [first["prompt"], second["prompt"]])
        self.assertEqual([(item["startFrame"], item["endFrame"]) for item in preset["timeline"]["segmentConfig"]["segments"]],
                         [(0, 240), (216, 456)])

    def test_a_split_the_timeline_cannot_honour_keeps_the_prompt_and_reports_it(self):
        draft = ("integrated_multimodal_description: [Shot 1] A woman returns a ticket.\n"
                 "overall_soundscape: Station ambience.\nnon_diegetic_music: N/A")
        with patch.object(self.nodes, "enhance_prompt", return_value=draft):
            result = self.nodes.MiniMaxH3PromptEnhancer.execute(**self.inputs(segment_count=2))
        legacy = (draft, "", "", "", 0, "")
        self.assertEqual(tuple(result.result)[:6], legacy)
        self.assertEqual(result.result[8], "")
        report = json.loads(result.result[6])
        self.assertEqual(report["status"], "not_split")
        self.assertEqual(report["segment_count"], 0)
        self.assertTrue(report["error"])
        self.assertEqual(result.result[7].strip(), report["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
