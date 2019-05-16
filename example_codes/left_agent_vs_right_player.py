from pathlib import Path
from time import sleep

import numpy as np
import pygame
import retro

if __name__ == '__main__' and __package__ is None:
    import sys, os
    sys.path.append(os.path.dirname(sys.path[0]))
    from utils import controller
    sys.path.pop()


ENV_NAME = 'MortalKombatII-Genesis'
STATE_PATH = Path("../states/2players_level1_LiuKangVs??.state")
DISPLAY_SCALE = 2


def init_env():
    env = retro.make(
        game=ENV_NAME,
        state=str(STATE_PATH.resolve()),
        players=2,
        use_restricted_actions=retro.Actions.ALL,
    )
    obs = env.reset()
    height, width, *_ = obs.shape
    output_dims = (width * DISPLAY_SCALE, height * DISPLAY_SCALE)

    display = pygame.display.set_mode(output_dims)

    return env, display


def render(disp, obs):
    height, width, *_ = obs.shape
    output_dims = (width * DISPLAY_SCALE, height * DISPLAY_SCALE)

    obs = np.transpose(obs, (1, 0, 2))
    surface = pygame.surfarray.make_surface(obs)
    pygame.transform.scale(surface, output_dims, disp)
    pygame.display.flip()

    pygame.event.pump()


# Dummy function
def get_agent_action(observation, act_space):
    return act_space.sample()[:12].tolist()


def main():
    if not STATE_PATH.is_file():
        print("Statefile does not exist!")
        return

    env, display = init_env()
    joy = controller.init_joystick()

    done = False
    obs = env.get_screen()
    while not done:
        render(display, obs)

        obs, _, done, _ = env.step(
            get_agent_action(obs, env.action_space) + controller.get_player_input(joy)
        )

        sleep(1 / 45)

if __name__ == '__main__':
    main()
