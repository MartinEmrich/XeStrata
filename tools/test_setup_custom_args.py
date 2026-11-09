# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""setup's -- separator: engine options after it are appended to the model's args, after setup's own.
No GPU or download needed: python tools/test_setup_custom_args.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup as S  # noqa: E402


class Split(unittest.TestCase):
    def test_split_at_the_first_double_dash(self):
        argv = ["--gpus", "0,1", "--no-start", "--", "--kv-resident", "20480"]
        self.assertEqual(S.split_custom_args(argv),
                         (["--gpus", "0,1", "--no-start"], ["--kv-resident", "20480"]))

    def test_without_the_separator_nothing_is_custom(self):
        self.assertEqual(S.split_custom_args(["--parallel", "2"]), (["--parallel", "2"], []))

    def test_the_separator_alone_is_empty(self):   # and a setup run then drops what an earlier one added
        self.assertEqual(S.split_custom_args(["--setup", "--"]), (["--setup"], []))

    def test_bench_original_compat_keeps_its_remainder(self):
        argv = ["--bench-original-compat", "--", "--long"]
        self.assertEqual(S.split_custom_args(argv), (argv, []))

    def test_a_second_double_dash_stays_custom(self):
        self.assertEqual(S.split_custom_args(["--no-start", "--", "--"]), (["--no-start"], ["--"]))


class Apply(unittest.TestCase):
    def test_applied_last_and_warned_about_duplicates(self):
        cfg = {"args": ["--pack", "p", "--kv", "fp16", "--mtp", "m"]}
        warns = S.apply_custom_args(cfg, ["--kv", "k8v4", "--batch-groups", "1"])
        self.assertEqual(cfg["args"], ["--pack", "p", "--kv", "fp16", "--mtp", "m",
                                       "--kv", "k8v4", "--batch-groups", "1"])
        self.assertEqual(len(warns), 1)
        self.assertIn("--kv", warns[0])
        self.assertNotIn("--batch-groups", warns[0])

    def test_empty_custom_changes_nothing(self):
        cfg = {"args": ["--pack", "p"]}
        self.assertEqual(S.apply_custom_args(cfg, []), [])
        self.assertEqual(cfg["args"], ["--pack", "p"])

    def test_a_flag_setup_does_not_name_needs_no_warning(self):
        cfg = {"args": ["--pack", "p"]}
        self.assertEqual(S.apply_custom_args(cfg, ["--conversation-save", "/var/lib/xestrata/conversations"]), [])
        self.assertEqual(cfg["args"][-2:], ["--conversation-save", "/var/lib/xestrata/conversations"])


if __name__ == "__main__":
    unittest.main()
