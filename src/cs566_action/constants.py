"""Dataset constants taken from the Penn Action annotations."""

JOINT_NAMES = (
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# These are the strings stored in the .mat files. Some names differ from the
# human-readable names in the dataset README (for example, "bowl" vs.
# "bowling"), so the annotation values are the canonical class identifiers.
ACTION_NAMES = (
    "baseball_pitch",
    "baseball_swing",
    "bench_press",
    "bowl",
    "clean_and_jerk",
    "golf_swing",
    "jump_rope",
    "jumping_jacks",
    "pullup",
    "pushup",
    "situp",
    "squat",
    "strum_guitar",
    "tennis_forehand",
    "tennis_serve",
)

TRAIN_FLAG = 1
TEST_FLAG = -1

