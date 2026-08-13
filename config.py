from pathlib import Path
#Change it according to your PC!
BASE_DIR = Path(r"C:\#Self Study\Projects\Casptone Prototype\src")
SEED = 42
#Keep it around 256 to 512 (aur badhyaa then RAM explosion lmao)
SEQ_LEN = 4
IMAGE_SIZE = 256
FEATURE_SIZE = 64


MIN_TARGETS = 0
MAX_TARGETS = 3

# Stationary stars and false moving-looking BLIPs
MIN_STARS = 40
MAX_STARS = 75
MIN_DISTRACTORS = 8
MAX_DISTRACTORS = 20

STAR_AMP_RANGE = (0.08, 0.60)
STAR_SIGMA_RANGE = (0.8, 2.0)
TARGET_AMP_RANGE = (0.65, 1.25)
TARGET_SIGMA_RANGE = (1.2, 2.2)
DISTRACTOR_AMP_RANGE = (0.45, 1.15)
DISTRACTOR_SIGMA_RANGE = (1.2, 2.2)

VELOCITY_RANGE = (4.0, 18.0)
MOTION_JITTER_STD = 0.15

#Asteroids Acceleartion
ACCELERATION_PROBABILITY = 0.05
ACCELERATION_RANGE = (-0.5, 0.5)

BACKGROUND_NOISE_STD_RANGE = (0.018, 0.040)

VERIFY_SAMPLES = 12
VERIFY_SEED = 777
VERIFY_OUTPUT_DIR = BASE_DIR / "generated_examples"

#Quick Defaults for quick run
TRAIN_SEQUENCES = 2000
VAL_SEQUENCES = 500
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
