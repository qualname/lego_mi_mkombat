import pygame

HIGH_KICK, LOW_KICK = 8, 0
PUNCH, BLOCK = 1, 3
MOVE_LEFT, MOVE_RIGHT = 6, 7
JUMP, SQUAT = 4, 5


def init_joystick():
    pygame.joystick.init()
    try:
        joy = pygame.joystick.Joystick(0)
        joy.init()
        return joy
    except pygame.error:
        return None


def _joy_state_to_action_space(joystick):
    pass


def _keyboard_state_to_action_space():
    """ Maps keyboard buttons to action values. """

    pushed_keys = [0] * 12
    keys = pygame.key.get_pressed()

    pushed_keys[JUMP] = keys[pygame.K_UP] or keys[pygame.K_w]
    pushed_keys[SQUAT] = keys[pygame.K_DOWN] or keys[pygame.K_s]
    pushed_keys[MOVE_LEFT] = keys[pygame.K_LEFT] or keys[pygame.K_a]
    pushed_keys[MOVE_RIGHT] = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    pushed_keys[HIGH_KICK] = keys[pygame.K_o]
    pushed_keys[LOW_KICK] = keys[pygame.K_l]
    pushed_keys[PUNCH] = keys[pygame.K_i]
    pushed_keys[BLOCK] = keys[pygame.K_u]

    return pushed_keys


def get_player_input(joy=None):
    if joy is not None:
        return _joy_state_to_action_space(joy)
    else:
        return _keyboard_state_to_action_space()
