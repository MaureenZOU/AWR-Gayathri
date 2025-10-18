import os
import math
import random
import pybullet as p
import json

# --------------------------------------------------------------------------- #
# Config --------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

JOINT_SAFETY_SCALAR = 0.95
N_SAMPLES           = 1          # number of poses to sample
_PRIMES             = [ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                        73, 79, 83, 89, 97, 101, 103, 107, 109,
                        113, 127, 131, 137, 139, 149, 151, 157,
                        163, 167, 173, 179, 181, 191, 193, 197,
                        199, 211, 223, 227, 229, 233 ]          # > enough for most hands

URDF_PATH  = "/disk1/data/awr/robot/xhand_right/urdf/xhand_right.urdf"
USE_EGL    = False           # GPU renderer
OUTPUT_DIR = "/disk1/data/awr/trellis/xhandr/joints_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Halton generator + Cranley–Patterson rotation ----------------------------- #
# --------------------------------------------------------------------------- #

def _radical_inverse(i, base):
    f, x = 1.0 / base, 0.0
    while i > 0:
        i, digit = divmod(i, base)
        x += digit * f
        f /= base
    return x

def _halton_sequence(n_pts, dim, seed=0):
    """
    Halton sequence in 'dim' dimensions, scrambled by a Cranley–Patterson rotation.
    The rotation is a per-dimension offset in [0,1) added modulo 1.
    """
    rng = random.Random(seed)                       # deterministic unless you change seed
    offsets = [rng.random() for _ in range(dim)]    # one offset per dimension

    seq = []
    for idx in range(1, n_pts + 1):                 # start at 1 so we never get 0
        pt = [ (_radical_inverse(idx, _PRIMES[d]) + offsets[d]) % 1.0
               for d in range(dim) ]
        seq.append(pt)
    return seq                                      # list of lists, values ∈ (0,1)

# --------------------------------------------------------------------------- #
# Pose computation (cached per loaded hand) --------------------------------- #
# --------------------------------------------------------------------------- #

_pose_cache = {}            # key = hand unique id, value = list[dict]

def _compute_poses(hand):
    n_jnts = p.getNumJoints(hand)

    # Low-discrepancy points (in unit cube), scrambled to avoid axis-alignment bias
    unit_points = _halton_sequence(N_SAMPLES, n_jnts, seed=0)  # change seed for a new set

    # Scale unit cube → (shrunk) joint limits
    lo_s, span_s = [], []
    for j in range(n_jnts):
        lo, hi = p.getJointInfo(hand, j)[8:10]
        if lo > hi:                               # continuous joint ⇒ −π … π
            lo, hi = -math.pi, math.pi
        span = hi - lo
        lo_s.append(lo + (1 - JOINT_SAFETY_SCALAR) * span * 0.5)
        span_s.append(JOINT_SAFETY_SCALAR * span)

    poses = []
    for u in unit_points:                         # u = list of unit coords
        pose = {}
        for j, u_j in enumerate(u):
            name = p.getJointInfo(hand, j)[1].decode()
            pose[name] = lo_s[j] + u_j * span_s[j]    # scale into real limits
        poses.append(pose)

    return poses

# --------------------------------------------------------------------------- #
# Public interface ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def sample_human_pose(hand, idx):
    """
    Return pose #idx (0 … N_SAMPLES-1) from a scrambled Halton sequence.

    Parameters
    ----------
    hand : int
        Body unique ID returned by p.loadURDF.
    idx  : int
        Which of the poses you want (0 <= idx < N_SAMPLES).
    """
    if hand not in _pose_cache:
        _pose_cache[hand] = _compute_poses(hand)

    if not (0 <= idx < N_SAMPLES):
        raise IndexError(f"idx must be 0 … {N_SAMPLES-1} (got {idx})")

    pose = _pose_cache[hand][idx]

    # put the hand into that configuration immediately
    for j in range(p.getNumJoints(hand)):
        name = p.getJointInfo(hand, j)[1].decode()
        p.resetJointState(hand, j, pose[name])

    return pose

# --------------------------------------------------------------------------- #
# Setup + sampling/export loop ---------------------------------------------- #
# --------------------------------------------------------------------------- #

if USE_EGL:
    p.connect(p.DIRECT)
    try:
        p.loadPlugin("eglRendererPlugin")
    except Exception:
        USE_EGL = False
else:
    p.connect(p.DIRECT)

p.setGravity(0, 0, -9.81)
hand = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0.1], useFixedBase=True)

for k in range(N_SAMPLES):
    joint_dict = sample_human_pose(hand, k)
    with open(os.path.join(OUTPUT_DIR, f"joint_states_{k:05d}.json"), "w") as f:
        json.dump(joint_dict, f, indent=2)
