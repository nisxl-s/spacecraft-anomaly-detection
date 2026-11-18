import unittest
from event_scoring import score_events


class EventScoringTest(unittest.TestCase):
    def test_clipping_and_boundary_overlap(self):
        r = score_events([(1, 4), (3, 7), (10, 13), (16, 17)], [(0, 5), (13, 16)], 5, 15)
        self.assertEqual(r['matches'], [(1, 0, 0), (2, 1, 3)])
        self.assertEqual(r['excluded_truths'], [0, 3])
        self.assertEqual(r['partly_clipped_truths'], [1])
        self.assertEqual(r['partly_clipped_alarms'], [0, 1])

    def test_cardinality_before_delay(self):
        # Alarm 0 can match either truth; alarm 1 matches only the first.
        r = score_events([(10, 20), (21, 25)], [(19, 21), (20, 20)], 0, 30)
        self.assertEqual(r['matches'], [(0, 1, 10), (1, 0, 0)])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (2, 0, 0))

    def test_delay_before_lexicographic_tie(self):
        r = score_events([(10, 30)], [(20, 20), (12, 12)], 0, 40)
        self.assertEqual(r['matches'], [(0, 1, 2)])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (1, 1, 0))

    def test_lex_tie_source_order(self):
        r = score_events([(10, 20), (10, 20)], [(12, 12), (12, 12)], 0, 30)
        self.assertEqual(r['matches'], [(0, 0, 2), (1, 1, 2)])

    def test_zero_cases(self):
        r = score_events([], [], 0, 10)
        self.assertIsNone(r['precision']); self.assertIsNone(r['recall'])
        r = score_events([(1, 2)], [], 0, 10)
        self.assertIsNone(r['precision']); self.assertEqual(r['recall'], 0)
        r = score_events([], [(1, 2)], 0, 10)
        self.assertEqual(r['precision'], 0); self.assertIsNone(r['recall'])

    def test_bridging_alarm_matches_only_once(self):
        r = score_events([(300, 305), (312, 316)], [(304, 314)], 250, 400)
        self.assertEqual(r['matches'], [(1, 0, 0)])
        self.assertEqual((r['tp'], r['fp'], r['fn']), (1, 0, 1))

if __name__ == '__main__':
    unittest.main()
