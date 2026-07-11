"""Validates `revision_merge.merge_fields` (and `merge_by_key`'s use of it):
during a scoped revision, a targeted entry's fields are spliced in ONE AT A
TIME — the model's new value for a field is accepted ONLY when it's
structurally different from the existing value; a field that's the same
(even if reworded to a differently-key-ordered-but-equal structure) keeps
its OLD value verbatim, so incidental LLM rewording of fields unrelated to
the actual edit can't silently drift.

Pure logic checks — no LLM/API calls, no I/O beyond stdout capture.
Run with: python -m unittest tests.test_revision_merge_fields -v
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from reel.revision_merge import merge_by_key, merge_fields


class TestMergeFields(unittest.TestCase):
    def test_unchanged_field_keeps_old_value(self):
        old = {"a": 1, "b": "same", "c": [1, 2, 3]}
        new = {"a": 99, "b": "same", "c": [1, 2, 3]}
        merged = merge_fields(old, new)
        self.assertEqual(merged, {"a": 99, "b": "same", "c": [1, 2, 3]})

    def test_unchanged_field_keeps_the_exact_old_object_not_a_copy(self):
        old_list = [1, 2, 3]
        old = {"a": old_list}
        new = {"a": [1, 2, 3]}   # equal content, a DIFFERENT list object
        merged = merge_fields(old, new)
        self.assertIs(merged["a"], old_list, "an unchanged field must keep the OLD object")

    def test_new_field_absent_from_old_is_taken_as_is(self):
        old = {"a": 1}
        new = {"a": 1, "b": "brand new"}
        merged = merge_fields(old, new)
        self.assertEqual(merged, {"a": 1, "b": "brand new"})

    def test_old_field_omitted_by_model_is_retained(self):
        old = {"a": 1, "b": "keep me"}
        new = {"a": 1}   # model's response simply didn't include "b"
        merged = merge_fields(old, new)
        self.assertEqual(merged, {"a": 1, "b": "keep me"})

    def test_dict_key_reordering_counts_as_unchanged(self):
        old = {"nested": {"x": 1, "y": 2}}
        new = {"nested": {"y": 2, "x": 1}}   # same content, different key order
        merged = merge_fields(old, new)
        self.assertIs(merged["nested"], old["nested"])

    def test_list_element_reordering_counts_as_changed(self):
        # List order is semantically meaningful (dialogue/shot/scene order) —
        # NOT tolerated the way dict key order is.
        old = {"items": ["a", "b", "c"]}
        new = {"items": ["c", "b", "a"]}
        merged = merge_fields(old, new)
        self.assertEqual(merged["items"], ["c", "b", "a"])

    def test_all_fields_different_matches_old_wholesale_replace_behavior(self):
        # When every field genuinely differs, the result must be identical
        # to what a wholesale entry replacement would have produced.
        old = {"a": 1, "b": 2}
        new = {"a": 10, "b": 20}
        self.assertEqual(merge_fields(old, new), new)

    def test_label_prints_changed_fields(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            merge_fields({"a": 1, "b": "same"}, {"a": 2, "b": "same"}, label="scene 3")
        out = buf.getvalue()
        self.assertIn("'scene 3'", out)
        self.assertIn("['a']", out)
        self.assertNotIn("'b'", out)

    def test_label_prints_no_change_when_nothing_differs(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            merge_fields({"a": 1}, {"a": 1}, label="scene 3")
        self.assertIn("kept entirely as-is", buf.getvalue())

    def test_no_label_prints_nothing(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            merge_fields({"a": 1}, {"a": 2})
        self.assertEqual(buf.getvalue(), "")


class TestMergeByKeyUsesFieldLevelMerge(unittest.TestCase):
    def test_targeted_entry_only_updates_the_fields_that_actually_differ(self):
        existing = [
            {"number": 1, "location": "Bar", "summary": "old summary", "purpose": "setup"},
            {"number": 2, "location": "Road", "summary": "s2", "purpose": "p2"},
        ]
        # Model's full response echoes scene 2 byte-identical (untouched,
        # not in revise_keys — merge_by_key's own outer splice handles that)
        # but for the TARGETED scene 1, only reworded "location"; "purpose"
        # is coincidentally re-emitted with identical text, "summary" is
        # genuinely reworded.
        new_partial = [
            {"number": 1, "location": "Tavern", "summary": "new summary", "purpose": "setup"},
        ]
        result = merge_by_key(existing, new_partial, lambda s: s["number"], {1})
        scene1 = next(s for s in result if s["number"] == 1)
        self.assertEqual(scene1["location"], "Tavern")     # genuinely changed -> new
        self.assertEqual(scene1["summary"], "new summary")  # genuinely changed -> new
        self.assertEqual(scene1["purpose"], "setup")         # coincidentally same -> old
        self.assertIs(scene1["purpose"], existing[0]["purpose"])
        # scene 2 untouched at the whole-entry level, as before
        scene2 = next(s for s in result if s["number"] == 2)
        self.assertIs(scene2, existing[1])

    def test_no_actual_field_difference_leaves_entry_effectively_unchanged(self):
        existing = [{"name": "Marcel", "physical_form": "tall, dark coat"}]
        # The model was asked to revise Marcel but its response happens to
        # be identical in substance (e.g. it decided nothing needed to change).
        new_partial = [{"name": "Marcel", "physical_form": "tall, dark coat"}]
        result = merge_by_key(existing, new_partial, lambda c: c["name"], {"Marcel"})
        self.assertEqual(result[0]["physical_form"], "tall, dark coat")
        self.assertIs(result[0]["physical_form"], existing[0]["physical_form"])


if __name__ == "__main__":
    unittest.main()
