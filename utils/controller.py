import pygame

HIGH_KICK, LOW_KICK = 8, 0
PUNCH, BLOCK = 1, 3
MOVE_LEFT, MOVE_RIGHT = 6, 7
JUMP, SQUAT = 4, 5


def _joy_state_to_action_space():
    pass


def _keyboard_state_to_action_space():
    pass


def get_player_input(joy=False):
    if joy:
        return _joy_state_to_action_space()
    else:
        return _keyboard_state_to_action_space()
