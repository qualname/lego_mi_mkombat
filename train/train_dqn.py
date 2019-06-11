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

    qnn = dqn.QNN(h, w, input_frames=config.BUNDLED_FRAMES, outputs=action_space.n).to(
        device
    )
    target_nn = dqn.QNN(
        h, w, input_frames=config.BUNDLED_FRAMES, outputs=action_space.n
    ).to(device)
    target_nn.load_state_dict(qnn.state_dict())
    target_nn.eval()

    memory = dqn.PrioritizedReplayMemory(
        max_len=config.BUFFER_LIMIT, batch_size=config.BATCH_SIZE
    )
    optimizer = torch.optim.Adam(qnn.parameters(), lr=config.LEARNING_RATE)

    return qnn, target_nn, memory, optimizer


def obs_to_gpu(screen):
    screen = screen.transpose((2, 0, 1))
    screen = numpy.ascontiguousarray(screen, dtype=numpy.float32) / 255
    screen = torch.from_numpy(screen)
    return screen.to(device)


def step(env, left_actions, right_actions=None, players=2):
    if players == 1:
        right_actions = numpy.ndarray((len(left_actions), 0))
    elif players == 2 and right_actions is None:
        right_actions = numpy.zeros_like(left_actions)

    if len(left_actions) == 1:
        left, right = left_actions[0], right_actions[0]
        observation, reward, done, info = env.step(left.tolist() + right.tolist())
        return obs_to_gpu(observation), reward, done, info

    states = (
        env.step(left.tolist() + right.tolist())
        for left, right in zip(left_actions, right_actions)
    )
    observations, rewards, dones, info = zip(*states)

    observations = torch.cat([obs_to_gpu(ob) for ob in observations], 0)
    if players == 2:
        reward = tuple(map(sum, zip(*rewards)))
    else:
        reward = tuple(sum(rewards), float('inf'))
    done = any(dones)

    return observations, reward, done, info[-1]


def main():
    player_count, left_char, right_char = get_state_info(config.STATE_PATH)

    env = retro.make(
        config.ENV_NAME,
        use_restricted_actions=retro.Actions.ALL,
        state=str(config.STATE_PATH.resolve()),
        players=player_count,
        scenario=str(config.SCENARIO_PATH.resolve()),
        info=str(config.INFO_PATH.resolve()),
    )
    env.reset()

    left_action_space = move_lists.ActionSpace(
        char_name=left_char,
        num_of_outputs=env.action_space.n // player_count,
        in_frames=config.BUNDLED_FRAMES,
    )

    qnn, target_nn, memory, optimizer = init(env.get_screen(), left_action_space)

    for episode in range(1, 10_000):
        temperature = config.MIN_EPSILON + (1.0 - config.MIN_EPSILON) / math.exp(
            episode / config.DECAY
        )

        env.reset()
        observation, *_, info = step(
            env, left_action_space.do_nothing(), players=player_count
        )

        # if the characters did no dmg to eachother in ~3930 frames
        # the game ends in a draw so we also stop the episode
        for _ in range(3920 // config.BUNDLED_FRAMES):
            action_id = qnn.sample_action(
                observation.unsqueeze(0), left_action_space, temperature
            )
            actions = left_action_space.to_action_list(action_id.item(), info)

            next_observation, reward, done, info = step(
                env, actions, players=player_count
            )
            reward = reward[0] if player_count == 2 else reward

            memory.push(
                (
                    observation.unsqueeze(0),
                    action_id,
                    torch.tensor([reward], device=device, dtype=torch.float),
                    next_observation.unsqueeze(0),
                    torch.tensor([done], device=device, dtype=torch.float),
                )
            )

            observation = next_observation

            if done:
                break

        if len(memory) > config.BUFFER_LIMIT // 10:
            beta = 0.4  # TODO: annealing 0.4 -> 1.0
            dqn.train(qnn, target_nn, memory, optimizer, beta)

        if episode % config.UPDATE_FREQ == 0:
            target_nn.load_state_dict(qnn.state_dict())

        # TODO: plot

    env.close()


if __name__ == '__main__':
    main()
