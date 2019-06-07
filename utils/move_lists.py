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


# TODO
_COMBOS = {'SubZero': []}


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
        self.char_name = char_name
        self.in_frames = in_frames
        self.n = len(_TO_IDX) + len(_COMBOS[char_name])
        self.orig_n = num_of_outputs

    def to_action_list(self, move_id: int) -> numpy.ndarray:
        act = numpy.zeros((self.in_frames, self.orig_n), dtype=numpy.int8)

        if move_id < len(_TO_IDX):
            idx = _TO_IDX[move_id]
            act[:, idx] = 1
        else:
            indices = COMBOS[self.char_name][move_id - len(_TO_IDX)]
            for i, idx in enumerate(indices):
                act[i, idx] = 1

        return act
