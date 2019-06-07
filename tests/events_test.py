from typing import List, Tuple

import pytest

from fran.events import EventLogger
from fran.common import Special


KEY = "a"


class Parametrizations:
    argnames = ("starts", "stops", "expecteds")

    def __init__(self):
        self.ids = []
        self.args = []

    def add(
        self,
        test_id: str,
        starts: List[int] = None,
        stops: List[int] = None,
        expecteds: List[Tuple[int, int]] = None,
    ):
        starts = starts or []
        stops = stops or []
        expecteds = expecteds or []

        self.ids.append(test_id)

        self.args.append((starts, stops, expecteds))


def _check(starts, stops, expecteds):
    el = EventLogger()

    for idx in starts:
        el.insert(KEY.lower(), idx)
    for idx in stops:
        el.insert(KEY.upper(), idx)

    assert list(el.start_stop_pairs(KEY.lower())) == expecteds


passing = Parametrizations()
passing.add("empty")
passing.add("simple", [10], [20], [(10, 20)])
passing.add("overlapping", [10, 15], [20, 25], [(10, 20), (15, 25)])
passing.add("start_stop_same", [10, 20], [20, 25], [(10, 20), (20, 25)])
passing.add("from_before", [], [20], [(Special.BEFORE, 20)])
passing.add("to_after", [10], [], [(10, Special.AFTER)])
passing.add("extra_starts", [5, 10], [20], [(5, 20), (10, Special.AFTER)])


@pytest.mark.parametrize(passing.argnames, passing.args, ids=passing.ids)
def test_pass(starts, stops, expecteds):
    _check(starts, stops, expecteds)


failing = Parametrizations()
failing.add("extra_stops", [10], [20, 25], [(Special.BEFORE, 20), (10, 25)])


@pytest.mark.parametrize(failing.argnames, failing.args, ids=failing.ids)
@pytest.mark.xfail(strict=True)
def test_fail(starts, stops, expecteds):
    _check(starts, stops, expecteds)
