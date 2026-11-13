"""Event -> H/A/D label derivation and the controlled-possession rule."""

import unittest

import numpy as np

from mtah.sample import A, D, H, X, controlled, event_labels


def ev(i, start, end, team, etype, subtype=""):
    return {"i": i, "start": start, "end": end, "team": team, "type": etype,
            "subtype": subtype, "period": 1}


def run(events, n=400):
    period = np.ones(n, dtype=np.int8)
    frames = np.arange(1, n + 1)
    return event_labels(events, period, n, lambda rows: frames[rows]), period, frames


class EventLabelTest(unittest.TestCase):
    def test_possession_follows_on_ball_events(self):
        lab, _, _ = run([ev(0, 1, 1, H, "SET PIECE", "KICK OFF"), ev(1, 1, 50, H, "PASS"),
                         ev(2, 100, 100, A, "RECOVERY"), ev(3, 100, 200, A, "PASS")])
        self.assertEqual(lab[0], H)
        self.assertEqual(lab[98], H)   # frame 99
        self.assertEqual(lab[99], A)   # frame 100
        self.assertEqual(lab[200], X)  # after the last event: ungraded

    def test_challenge_does_not_change_possession(self):
        lab, _, _ = run([ev(0, 1, 50, H, "PASS"), ev(1, 60, 60, A, "CHALLENGE", "AERIAL-WON"),
                         ev(2, 80, 120, H, "PASS")])
        self.assertTrue(np.all(lab[:120] == H))

    def test_ball_out_is_dead_until_restart(self):
        lab, _, _ = run([ev(0, 1, 40, H, "PASS"), ev(1, 40, 60, H, "BALL OUT"),
                         ev(2, 40, 41, H, "CARRY"),  # zero-length carry listed after BALL OUT
                         ev(3, 150, 150, A, "SET PIECE", "THROW IN"), ev(4, 150, 200, A, "PASS")])
        self.assertEqual(lab[58], H)   # frame 59, still in flight
        self.assertEqual(lab[59], D)   # frame 60, out
        self.assertEqual(lab[148], D)
        self.assertEqual(lab[149], A)  # throw-in

    def test_restart_implies_stoppage(self):
        lab, _, _ = run([ev(0, 1, 40, H, "PASS"), ev(1, 40, 70, H, "BALL LOST", "FORCED"),
                         ev(2, 150, 150, H, "SET PIECE", "FREE KICK"), ev(3, 150, 200, H, "PASS")])
        self.assertEqual(lab[68], H)
        self.assertEqual(lab[69], D)   # from the end of the last on-ball action
        self.assertEqual(lab[149], H)

    def test_foul_starts_dead_ball(self):
        lab, _, _ = run([ev(0, 1, 40, H, "PASS"), ev(1, 50, 50, A, "CHALLENGE",
                                                    "TACKLE-FAULT-LOST"),
                         ev(2, 120, 120, H, "SET PIECE", "FREE KICK"), ev(3, 120, 200, H, "PASS")])
        self.assertEqual(lab[48], H)
        self.assertEqual(lab[49], D)


class ControlledTest(unittest.TestCase):
    def test_short_spell_without_passes_reverts(self):
        events = [ev(0, 1, 100, H, "PASS"), ev(1, 101, 101, A, "RECOVERY"),
                  ev(2, 120, 120, H, "RECOVERY"), ev(3, 120, 300, H, "PASS")]
        lab, period, frames = run(events)
        self.assertTrue((lab == A).any())
        out = controlled(lab, lambda rows: frames[rows], events, period)
        self.assertFalse((out == A).any())

    def test_long_spell_is_kept(self):
        events = [ev(0, 1, 100, H, "PASS"), ev(1, 101, 101, A, "RECOVERY"),
                  ev(2, 250, 250, H, "RECOVERY"), ev(3, 250, 350, H, "PASS")]
        lab, period, frames = run(events)
        out = controlled(lab, lambda rows: frames[rows], events, period)
        np.testing.assert_array_equal(out, lab)

    def test_spell_after_restart_is_kept(self):
        events = [ev(0, 1, 50, H, "BALL OUT"), ev(1, 100, 100, A, "SET PIECE", "THROW IN"),
                  ev(2, 110, 110, H, "RECOVERY"), ev(3, 110, 300, H, "PASS")]
        lab, period, frames = run(events)
        out = controlled(lab, lambda rows: frames[rows], events, period)
        np.testing.assert_array_equal(out, lab)


if __name__ == "__main__":
    unittest.main()
