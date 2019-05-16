import retro

from pathlib import Path
from time import sleep


env_name = 'MortalKombatII-Genesis'
state_name = '2players_level1_SubZeroVsJax'
state_path = Path(f'../states/{state_name}.state').resolve()


env = retro.make(
    env_name,
    use_restricted_actions=retro.Actions.ALL,
    state=str(state_path.resolve()),
    players=2
)
state = env.reset()

while True:
    done = False
    while not done:
        observation, reward, done, info = env.step(
            f'{128|2048:012b}' + f'{128|2048:012b}'   #Player1 Action + Player2 Action
        )
        env.render()
        sleep(1/45)

    env.reset()
