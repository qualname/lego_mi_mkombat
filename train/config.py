from pathlib import Path

ENV_NAME = 'MortalKombatII-Genesis'
STATE_PATH = Path('../states/2players_level1_SubZeroVsJax.state')
SCENARIO_PATH = Path('scenario.json')

BUFFER_LIMIT = 10_000
BATCH_SIZE = 128
LEARNING_RATE = 0.001

DECAY = 2000
MIN_EPSILON = 0.005
