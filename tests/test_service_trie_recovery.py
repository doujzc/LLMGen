from __future__ import annotations

import unittest

import service
import service_910b
import service_openai


SERVICE_MODULES = (service, service_openai, service_910b)


class MultiPathTokenTrieRecoveryTest(unittest.TestCase):
    def _tries(self):
        for module in SERVICE_MODULES:
            with self.subTest(module=module.__name__):
                yield module.MultiPathTokenTrie(
                    ((1, 2), (1, 3), (4, 5)),
                    eos_token_id=0,
                    separator_token_ids=(9,),
                    max_paths=3,
                )

    def test_strict_output_is_not_marked_recovered(self) -> None:
        for trie in self._tries():
            result = trie.parse_with_recovery((1, 2, 9, 4, 5))
            self.assertEqual(result.paths, ((1, 2), (4, 5)))
            self.assertFalse(result.recovered)
            self.assertEqual(result.consumed_tokens, 5)
            self.assertEqual(result.discarded_tokens, 0)
            self.assertIsNone(result.reason)

    def test_invalid_suffix_recovers_last_complete_path(self) -> None:
        for trie in self._tries():
            result = trie.parse_with_recovery((1, 2, 5))
            self.assertEqual(result.paths, ((1, 2),))
            self.assertTrue(result.recovered)
            self.assertEqual(result.consumed_tokens, 2)
            self.assertEqual(result.discarded_tokens, 1)
            self.assertEqual(result.reason, "invalid_path_boundary")
            with self.assertRaises(RuntimeError):
                trie.parse_complete((1, 2, 5))

    def test_incomplete_second_path_keeps_first_path(self) -> None:
        for trie in self._tries():
            result = trie.parse_with_recovery((1, 2, 9, 4))
            self.assertEqual(result.paths, ((1, 2),))
            self.assertTrue(result.recovered)
            self.assertEqual(result.consumed_tokens, 2)
            self.assertEqual(result.discarded_tokens, 2)
            self.assertEqual(result.reason, "incomplete_path")

    def test_duplicate_path_suffix_keeps_first_path(self) -> None:
        for trie in self._tries():
            result = trie.parse_with_recovery((1, 2, 9, 1, 2))
            self.assertEqual(result.paths, ((1, 2),))
            self.assertTrue(result.recovered)
            self.assertEqual(
                result.reason,
                "invalid_or_duplicate_path_prefix",
            )

    def test_no_complete_prefix_remains_a_failure(self) -> None:
        for trie in self._tries():
            result = trie.parse_with_recovery((1, 8))
            self.assertEqual(result.paths, ())
            self.assertFalse(result.recovered)
            self.assertEqual(result.consumed_tokens, 0)
            self.assertEqual(result.discarded_tokens, 2)

    def test_request_parser_does_not_scan_candidate_collection(self) -> None:
        class NoCandidateIteration:
            def __init__(self, size: int) -> None:
                self.size = size

            def __len__(self) -> int:
                return self.size

            def __iter__(self):
                raise AssertionError("request parsing scanned every candidate")

        for trie in self._tries():
            trie.paths = NoCandidateIteration(3)
            self.assertEqual(trie.allowed_next((1,)), (2, 3))
            self.assertEqual(
                trie.parse_with_recovery((1, 2, 5)).paths,
                ((1, 2),),
            )


if __name__ == "__main__":
    unittest.main()
