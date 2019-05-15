import pygame

_HIGH_KICK, _LOW_KICK = 8, 0
_PUNCH, _BLOCK = 1, 3
_MOVE_LEFT, _MOVE_RIGHT = 6, 7
_JUMP, _SQUAT = 4, 5


def init_joystick():
    pygame.joystick.init()
    try:
        joy = pygame.joystick.Joystick(0)
        joy.init()
        return joy
    except pygame.error:
        return None


def _get_movement_state(joystick):
    """ Returns movement-related `x` and `y` states.
    Prefers 4 arrow buttons over thumbstick. """

    x, y = joystick.get_hat(0)
    if x == 0:
        x = joystick.get_axis(0)
    if y == 0:
        y = joystick.get_axis(1) * -1

    return x, y


def _joy_state_to_action_space(joystick):
    """ Maps joystick buttons to action values. """

    button_to_action_map = {
        0: _HIGH_KICK,  # △
        1: _PUNCH,      # ◯
        2: _LOW_KICK,   # ✕
        3: _BLOCK,      # □
    }

    pushed_keys = [0] * 12

    for button, action in button_to_action_map.items():
        pushed_keys[action] = joystick.get_button(button)

    x, y = _get_movement_state(joystick)

    if x < -0.5:
        pushed_keys[_MOVE_LEFT] = 1
        pushed_keys[_MOVE_RIGHT] = 0
    elif x > 0.5:
        pushed_keys[_MOVE_LEFT] = 0
        pushed_keys[_MOVE_RIGHT] = 1
    else:
        pushed_keys[_MOVE_LEFT] = 0
        pushed_keys[_MOVE_RIGHT] = 0

    if y < -0.5:
        pushed_keys[_JUMP] = 0
        pushed_keys[_SQUAT] = 1
    elif y > 0.5:
        pushed_keys[_JUMP] = 1
        pushed_keys[_SQUAT] = 0
    else:
        pushed_keys[_JUMP] = 0
        pushed_keys[_SQUAT] = 0

    return pushed_keys


def _keyboard_state_to_action_space():
    """ Maps keyboard buttons to action values. """

    pushed_keys = [0] * 12
    keys = pygame.key.get_pressed()

    pushed_keys[_JUMP] = keys[pygame.K_UP] or keys[pygame.K_w]
    pushed_keys[_SQUAT] = keys[pygame.K_DOWN] or keys[pygame.K_s]
    pushed_keys[_MOVE_LEFT] = keys[pygame.K_LEFT] or keys[pygame.K_a]
    pushed_keys[_MOVE_RIGHT] = keys[pygame.K_RIGHT] or keys[pygame.K_d]

    pushed_keys[_HIGH_KICK] = keys[pygame.K_o]
    pushed_keys[_LOW_KICK] = keys[pygame.K_l]
    pushed_keys[_PUNCH] = keys[pygame.K_i]
    pushed_keys[_BLOCK] = keys[pygame.K_u]

    return pushed_keys


def get_player_input(joy=None):
    if joy is not None:
        return _joy_state_to_action_space(joy)
    else:
        return _keyboard_state_to_action_space()
