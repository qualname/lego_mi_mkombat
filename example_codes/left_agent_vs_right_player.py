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
STATE_PATH = Path('../states/2players_level1_LiuKangVs??.state')


def init_env():
    env = retro.make(
        game=ENV_NAME,
        state=str(STATE_PATH.resolve()),
        players=2,
        use_restricted_actions=retro.Actions.ALL,
    )
    _ = env.reset()

    return env


# Dummy function
def get_agent_action(observation, act_space):
    return act_space.sample()[:12].tolist()


def main():
    if not STATE_PATH.is_file():
        print('Statefile does not exist!')
        return

    env = init_env()
    display = renderer.init_display(initial_obs=env.get_screen())

    joy = controller.init_joystick()

    done = False
    obs = env.get_screen()
    while not done:
        renderer.render(display, obs)

        obs, _, done, _ = env.step(
            get_agent_action(obs, env.action_space) + controller.get_player_input(joy)
        )

        sleep(1 / 45)


if __name__ == '__main__':
    main()
