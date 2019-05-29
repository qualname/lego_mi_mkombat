import retro
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


def main():
    player_count, left_char, right_char = get_state_info(config.STATE_PATH)

    env = retro.make(
        config.ENV_NAME,
        use_restricted_actions=retro.Actions.ALL,
        state=str(config.STATE_PATH.resolve()),
        players=player_count,
        scenario=config.SCENARIO_PATH,
    )
    env.reset()

    left_action_space = move_lists.ActionSpace(left_char, env.action_space)
    # right_action_space = move_lists.ActionSpace(right_char, env.action_space)

    qnn, memory, optimizer = init(env.get_screen(), left_action_space)

    env.close()
