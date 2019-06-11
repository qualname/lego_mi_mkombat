import numpy

from .controller import (
    _HIGH_KICK,
    _LOW_KICK,
    _PUNCH,
    _BLOCK,
    _MOVE_LEFT,
    _MOVE_RIGHT,
    _JUMP,
    _SQUAT,
)

FORWARD = _MOVE_RIGHT  # TODO
BACK = _MOVE_LEFT      # TODO
LOW_PUNCH = _PUNCH
HIGH_PUNCH = _PUNCH

_COMBOS = {
    'SubZero': [
        [_SQUAT, FORWARD, [FORWARD, LOW_PUNCH]],
        [_SQUAT, BACK, [BACK, _LOW_KICK]],
        [[_LOW_KICK, _HIGH_KICK, BACK]],
    ],
}


_TO_IDX = {
    0: _HIGH_KICK,
    1: _LOW_KICK,
    2: _PUNCH,
    3: _BLOCK,
    4: _MOVE_LEFT,
    5: _MOVE_RIGHT,
    6: _JUMP,
    7: _SQUAT,
}


class ActionSpace:
    """ A character-specific action space.

    The MortalKombat gym environment's `step(...)` function expects a MultiBinary input.
    DQNs output/predict a single action.
    Therefore, this custom action space is used to map SingleDiscrete outputs to
    MultiBinary.
    """

    def __init__(self, char_name: str, num_of_outputs: int, in_frames: int) -> None:
        assert char_name in _COMBOS.keys()
        assert in_frames >= len(max(_COMBOS[char_name], key=len))

        self.char_name = char_name
        self.in_frames = in_frames
        self.n = len(_TO_IDX) + len(_COMBOS[char_name])
        self.orig_n = num_of_outputs

        self.facing_right = True

    def to_action_list(self, move_id: int, info: dict) -> numpy.ndarray:
        diff = info['enemy_x_position'] - info['x_position']
        self.facing_right = diff > 0

        act = numpy.zeros((self.in_frames, self.orig_n), dtype=numpy.int8)

        if move_id < len(_TO_IDX):
            idx = _TO_IDX[move_id]
            act[:, idx] = 1
        else:
            indices = _COMBOS[self.char_name][move_id - len(_TO_IDX)]
            for i, idx in enumerate(indices):
                idx = self.resolve_forwards_and_backs(idx)
                act[i, idx] = 1

        return act

    def do_nothing(self) -> numpy.ndarray:
        return numpy.zeros((self.in_frames, self.orig_n), dtype=numpy.int8)

    def resolve_forwards_and_backs(self, command):
        if isinstance(command, list):
            if FORWARD in command:
                idx = command.index(FORWARD)
                command[idx] = _MOVE_RIGHT if self.facing_right else _MOVE_LEFT
            elif BACK in command:
                idx = command.index(BACK)
                command[idx] = _MOVE_LEFT if self.facing_right else _MOVE_RIGHT

        else:
            if FORWARD == command:
                command = _MOVE_RIGHT if self.facing_right else _MOVE_LEFT
            elif BACK == command:
                command = _MOVE_LEFT if self.facing_right else _MOVE_RIGHT

        return command
