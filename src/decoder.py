"""Sequence decoder: the security state machine that arms the system.

Pure Python (no OpenCV): it receives one label per frame and the current time,
so it can be tested with simulated sequences and ported to other languages.

States
    LOCKED    waiting for the first card
    ENTERING  1..3 cards stored, waiting for the next one
    VERIFY    4 cards stored: compared with the secret code (transient)
    ARMED     correct code: the tracker and turret run
    PENALTY   too many wrong codes: every card is ignored for a while

Reading a card (debounce)
    The detector output flickers (missed frames, a hand passing by), so a card
    is only "read" when the same label fills at least agree_frac of the last
    window_frames frames (default 6 of 10: a majority, so two symbols can never
    both qualify, and up to 40 % of lost frames are tolerated). After a read, the
    card must be REMOVED before the next read: holding a card counts once, and
    showing the same symbol twice in a row requires taking it away and back.
    Hysteresis: reading needs 6 of 10 frames, but removal needs the WHOLE window
    empty (~0.33 s without a card). Otherwise, while a card is held, a burst of
    missed detections would look like a removal and the card would be read twice.

Security rules
    * The code is only checked after all 4 cards: an early rejection would tell
      an attacker which position is wrong (it would turn 7^4 guesses into 4*7).
    * CANCEL clears the cards entered so far; 'r' (reset) does the same and
      disarms when ARMED, but it never shortens a PENALTY.
    * timeout_s without a new card clears the partial sequence.
    * max_failures wrong codes in a row -> PENALTY, with escalating durations
      (penalties_s, e.g. 10/20/40 s). A correct code resets the counters.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

LOCKED = "LOCKED"
ENTERING = "ENTERING"
VERIFY = "VERIFY"
ARMED = "ARMED"
PENALTY = "PENALTY"

CANCEL = "CANCEL"
UNKNOWN = "UNKNOWN"
VALID_SYMBOLS = {"CIRCLE", "BULLSEYE", "TRIANGLE", "DIAMOND", "PENTAGON", "BARS", "DOTS"}
CODE_MAX_LEN = 4


def select_label(cards: list[tuple[str, float]]) -> str | None:
    """Label of the frame from the detected cards [(label, area), ...].

    UNKNOWN cards are ignored. If two different known symbols are visible at the
    same time the frame is ambiguous and nothing is read (cards must be shown
    one by one).
    """
    known = {label for label, _ in cards if label != UNKNOWN}
    return known.pop() if len(known) == 1 else None


@dataclass
class Event:
    t: float
    name: str        # read, cancel, timeout, reset, wrong_code, armed, penalty, penalty_end, disarmed
    detail: str = ""


class SequenceDecoder:
    def __init__(self, code: list[str], window_frames: int = 10, agree_frac: float = 0.6,
                 timeout_s: float = 8.0, max_failures: int = 3,
                 penalties_s: list[float] | tuple[float, ...] = (10.0, 20.0, 40.0)):
        code = [str(c).upper() for c in code]
        if not 1 <= len(code) <= CODE_MAX_LEN:
            raise ValueError(f"The code must have 1 to {CODE_MAX_LEN} symbols")
        bad = [c for c in code if c not in VALID_SYMBOLS]
        if bad:
            raise ValueError(f"Invalid symbols in the code: {bad} (valid: {sorted(VALID_SYMBOLS)})")
        self.code = code
        self.window = deque(maxlen=window_frames)
        self.need = max(1, int(round(agree_frac * window_frames)))
        self.timeout_s = timeout_s
        self.max_failures = max_failures
        self.penalties_s = list(penalties_s)
        self.events: list[Event] = []
        self.state = LOCKED
        self.entered: list[str] = []
        self.failures = 0          # wrong codes since the last penalty or success
        self.penalty_level = 0     # how many penalties so far (escalates the duration)
        self.awaiting_release = False
        self.last_read_t = 0.0
        self.penalty_until = 0.0
        self.message = "Show the first card"

    @classmethod
    def from_config(cls, cfg: dict) -> "SequenceDecoder":
        d = cfg.get("decoder", {})
        return cls(d["code"], d.get("window_frames", 10), d.get("agree_frac", 0.6), d.get("timeout_s", 8.0),
                   d.get("max_failures", 3), d.get("penalties_s", (10.0, 20.0, 40.0)))

    # ------------------------------------------------------------------ helpers
    def _log(self, t: float, name: str, detail: str = "") -> None:
        self.events.append(Event(t, name, detail))

    def _clear(self) -> None:
        self.entered = []
        self.state = LOCKED

    def _stable(self) -> str | None:
        """Label that fills the window (or None if no label reaches agree_frac)."""
        if len(self.window) < self.window.maxlen:
            return None
        label, n = Counter(self.window).most_common(1)[0]
        return label if n >= self.need else None

    def _released(self) -> bool:
        """Card removed: the whole window without a card (hysteresis, see module docstring)."""
        return len(self.window) == self.window.maxlen and all(lab is None for lab in self.window)

    @property
    def armed(self) -> bool:
        return self.state == ARMED

    def hold_progress(self) -> tuple[str | None, float]:
        """Card currently being held and how close it is to being read (0..1), for the UI."""
        if self.awaiting_release or not self.window or self.state in (ARMED, PENALTY):
            return None, 0.0
        label, n = Counter(lab for lab in self.window if lab is not None).most_common(1)[0] \
            if any(lab is not None for lab in self.window) else (None, 0)
        return label, min(1.0, n / self.need)

    def remaining(self, now: float) -> float:
        """Seconds left of the penalty, or until the partial sequence times out."""
        if self.state == PENALTY:
            return max(0.0, self.penalty_until - now)
        if self.state == ENTERING:
            return max(0.0, self.last_read_t + self.timeout_s - now)
        return 0.0

    # ------------------------------------------------------------------ inputs
    def reset(self, now: float) -> None:
        """'r' key: forget the partial sequence or disarm. It does not end a penalty."""
        if self.state == PENALTY:
            self.message = "Reset ignored during the penalty"
            return
        if self.state == ARMED:
            self._log(now, "disarmed")
        else:
            self._log(now, "reset")
        self._clear()
        self.window.clear()
        self.awaiting_release = False
        self.message = "Reset: show the first card"

    def update(self, label: str | None, now: float) -> str:
        """Feed the label of one frame (None = no card). Returns the current state."""
        label = None if label in (None, UNKNOWN) else str(label).upper()

        if self.state == ARMED:
            return self.state

        if self.state == PENALTY:
            if now < self.penalty_until:
                return self.state
            self._log(now, "penalty_end")
            self.state = LOCKED
            self.window.clear()
            self.awaiting_release = False
            self.message = "Penalty over: show the first card"

        # Partial sequence abandoned for too long
        if self.state == ENTERING and now - self.last_read_t > self.timeout_s:
            self._log(now, "timeout", f"{len(self.entered)} cards")
            self._clear()
            self.message = "Timeout: sequence cleared"

        self.window.append(label)
        if self.awaiting_release:
            if self._released():
                self.awaiting_release = False
            return self.state

        read = self._stable()
        if read is None:
            return self.state

        # A card has been read
        self.awaiting_release = True
        self.last_read_t = now
        if read == CANCEL:
            self._log(now, "cancel", f"{len(self.entered)} cards")
            self._clear()
            self.message = "Cancelled: sequence cleared"
            return self.state

        self.entered.append(read)
        self._log(now, "read", read)
        self.state = ENTERING
        self.message = f"Card {len(self.entered)}/{len(self.code)} stored: remove it"
        if len(self.entered) == len(self.code):
            self._verify(now)
        return self.state

    def _verify(self, now: float) -> None:
        self.state = VERIFY
        if self.entered == self.code:
            self._log(now, "armed")
            self.state = ARMED
            self.failures = 0
            self.penalty_level = 0
            self.entered = []
            self.message = "Access granted ('r' disarms)"
            return
        self.failures += 1
        self._log(now, "wrong_code", f"attempt {self.failures}/{self.max_failures}")
        self._clear()
        if self.failures >= self.max_failures:
            dur = self.penalties_s[min(self.penalty_level, len(self.penalties_s) - 1)]
            self.penalty_level += 1
            self.failures = 0
            self.state = PENALTY
            self.penalty_until = now + dur
            self._log(now, "penalty", f"{dur:g} s")
            self.message = f"Too many attempts: {dur:g} s"
        else:
            self.message = f"Wrong code ({self.failures}/{self.max_failures})"
