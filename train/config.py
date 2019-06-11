from pathlib import Path

ENV_NAME = 'MortalKombatII-Genesis'
STATE_PATH = Path('../states/2players_level1_SubZeroVsJax.state')
SCENARIO_PATH = Path('scenario.json')
INFO_PATH = Path('data.json')


BUFFER_LIMIT = 10_000
BATCH_SIZE = 128
LEARNING_RATE = 0.001

BUNDLED_FRAMES = 4

DECAY = 2000
MIN_EPSILON = 0.005

UPDATE_FREQ = 10
