import math

import retro
import numpy
import torch

if __name__ == '__main__' and __package__ is None:
    import sys, os
    sys.path.append(os.path.dirname(sys.path[0]))

import config
from models import dqn
from utils import move_lists

if __name__ == '__main__' and __package__ is None:
    sys.path.pop()


if not config.STATE_PATH.is_file():
    raise ValueError(f'Invalid state path ({config.STATE_PATH})!')
if not config.SCENARIO_PATH.is_file():
    raise ValueError(f'Invalid scenario path ({config.SCENARIO_PATH})!')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_state_info(state_path):
    file_name, _ = state_path.name.rsplit(sep='.', maxsplit=1)
    player_cnt, _, characters = file_name.split('_')

    player_cnt = int(player_cnt[0])
    assert player_cnt in (1, 2)

    left, right = characters.split('Vs')

    return player_cnt, left, right


def init(init_observation, action_space):
    h, w, *_ = init_observation.shape

    qnn = dqn.QNN(h, w, input_frames=1, outputs=action_space.n).to(device)

    memory = dqn.ReplayMemory(max_len=config.BUFFER_LIMIT, batch_size=config.BATCH_SIZE)
    optimizer = torch.optim.Adam(qnn.parameters(), lr=config.LEARNING_RATE)

    return qnn, memory, optimizer


def obs_to_gpu(screen):
    screen = screen.transpose((2, 0, 1))
    screen = numpy.ascontiguousarray(screen, dtype=numpy.float32) / 255
    screen = torch.from_numpy(screen)
    return screen.unsqueeze(0).to(device)


def main():
    player_count, left_char, right_char = get_state_info(config.STATE_PATH)

    env = retro.make(
        config.ENV_NAME,
        use_restricted_actions=retro.Actions.ALL,
        state=str(config.STATE_PATH.resolve()),
        players=player_count,
        scenario=str(config.SCENARIO_PATH.resolve()),
    )
    env.reset()

    left_action_space = move_lists.ActionSpace(left_char, env.action_space, 1)
    # right_action_space = move_lists.ActionSpace(right_char, env.action_space, 1)

    qnn, memory, optimizer = init(env.get_screen(), left_action_space)

    for episode in range(10_000):
        temperature = config.MIN_EPSILON + (1.0 - config.MIN_EPSILON) / math.exp(
            episode / config.DECAY
        )

        observation = env.reset()
        observation = obs_to_gpu(observation)

        done = False
        while not done:
            action_id = qnn.sample_action(observation, left_action_space, temperature)
            actions = left_action_space.to_action_list(action_id.item())

            next_observation, reward, done, _ = env.step(actions[0].tolist() + [0] * 12)
            reward = reward[0] if player_count == 2 else reward
            next_observation = obs_to_gpu(next_observation)

            memory.push(
                (
                    observation,
                    action_id,
                    torch.tensor([reward], device=device, dtype=torch.float),
                    next_observation,
                    torch.tensor([done], device=device, dtype=torch.float),
                )
            )

            observation = next_observation

            if len(memory) > config.BUFFER_LIMIT // 10:
                dqn.train(qnn, memory, optimizer)

        # TODO: plot

    env.close()


if __name__ == '__main__':
    main()
