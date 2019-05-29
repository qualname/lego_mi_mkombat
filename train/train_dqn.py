import retro

import config
try:
    from ..utils import move_lists
except ValueError:
    import sys, os
    sys.path.append(os.path.dirname(sys.path[0]))
    from utils import move_lists


if not config.STATE_PATH.is_file():
    raise ValueError(f'Invalid state path ({config.STATE_PATH})!')
if not config.SCENARIO_PATH.is_file():
    raise ValueError(f'Invalid scenario path ({config.SCENARIO_PATH})!')


def get_state_info(state_path):
    file_name, _ = state_path.name.rsplit(sep='.', maxsplit=1)
    player_cnt, _, characters = file_name.split('_')

    player_cnt = int(player_cnt[0])
    assert player_cnt in (1, 2)

    left, right = characters.split('Vs')

    return player_cnt, left, right


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

    env.close()
