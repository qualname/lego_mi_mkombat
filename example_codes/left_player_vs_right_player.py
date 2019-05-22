from pathlib import Path
from time import sleep

import retro

# fmt: off
if __name__ == '__main__' and __package__ is None:
    import sys, os
    sys.path.append(os.path.dirname(sys.path[0]))
    from utils import controller
    from utils import renderer
    sys.path.pop()
# fmt: on


ENV_NAME = 'MortalKombatII-Genesis'
STATE_PATH = Path('../states/2players_level1_??Vs??.state')


def init_env():
    env = retro.make(
        game=ENV_NAME,
        state=str(STATE_PATH.resolve()),
        players=2,
        use_restricted_actions=retro.Actions.ALL,
    )
    _ = env.reset()

    return env


def main():
    if not STATE_PATH.is_file():
        print('Statefile does not exist!')
        return

    env = init_env()
    display = renderer.init_display(initial_obs=env.get_screen())

    left = controller.init_joystick(0)
    right = controller.init_joystick(1)

    done = False
    obs = env.get_screen()
    while not done:
        renderer.render(display, obs)

        obs, _, done, _ = env.step(
            controller.get_player_input(left) + controller.get_player_input(right)
        )

        sleep(1 / 45)


if __name__ == '__main__':
    main()
