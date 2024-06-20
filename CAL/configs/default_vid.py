import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root path for dataset directory
_C.DATA.ROOT = '/data/CUI/briar/data_working/dawei.du/datasets'
# _C.DATA.ROOT = '../../mevid'#'../datasets'#
# Dataset for evaluation
_C.DATA.DATASET = 'ccvid'
# Whether split each full-length video in the training set into some clips
_C.DATA.DENSE_SAMPLING = False
# Sampling step of dense sampling for training set
_C.DATA.SAMPLING_STEP = 64
# Workers for dataloader
_C.DATA.NUM_WORKERS = 4
# Height of input image
_C.DATA.HEIGHT = 256
# Width of input image
_C.DATA.WIDTH = 128
# Batch size for training
_C.DATA.TRAIN_BATCH = 64
# Batch size for testing
_C.DATA.TEST_BATCH = 128
# The number of instances per identity for training sampler
_C.DATA.NUM_INSTANCES = 1
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Random erase prob
_C.AUG.RE_PROB = 0.0
# Temporal sampling mode for training, 'tsn' or 'stride'
_C.AUG.TEMPORAL_SAMPLING_MODE = 'stride'
# Sequence length of each input video clip
_C.AUG.SEQ_LEN = 8
# Sampling stride of each input video clip
_C.AUG.SAMPLING_STRIDE = 4
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name. All supported model can be seen in models/__init__.py
_C.MODEL.NAME = 'c2dres50'
# The stride for laery4 in resnet
_C.MODEL.RES4_STRIDE = 1
# feature dim
_C.MODEL.FEATURE_DIM = 2048
# Model path for resuming
_C.MODEL.RESUME = ''
# Params for AP3D
_C.MODEL.AP3D = CN()
# Temperature for APM
_C.MODEL.AP3D.TEMPERATURE = 4
# Contrastive attention
_C.MODEL.AP3D.CONTRACTIVE_ATT = True
# -----------------------------------------------------------------------------
# Losses for training 
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Classification loss
_C.LOSS.CLA_LOSS = 'crossentropy'
# Clothes classification loss
_C.LOSS.CLOTHES_CLA_LOSS = 'cosface'
# Scale for classification loss
_C.LOSS.CLA_S = 16.
# Margin for classification loss
_C.LOSS.CLA_M = 0.
# Pairwise loss
_C.LOSS.PAIR_LOSS = 'triplet'
# The weight for pairwise loss
_C.LOSS.PAIR_LOSS_WEIGHT = 0.0
# Scale for pairwise loss
_C.LOSS.PAIR_S = 16.
# Margin for pairwise loss
_C.LOSS.PAIR_M = 0.3
# Clothes-based adversarial loss
_C.LOSS.CAL = 'cal'
# Epsilon for clothes-based adversarial loss
_C.LOSS.EPSILON = 0.1
# Momentum for clothes-based adversarial loss with memory bank
_C.LOSS.MOMENTUM = 0.
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 150
# Start epoch for clothes classification
_C.TRAIN.START_EPOCH_CC = 50
# Start epoch for adversarial training
_C.TRAIN.START_EPOCH_ADV = 50
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 0.00035
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# Stepsize to decay learning rate
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [40, 80, 120]
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# Using amp for training
_C.TRAIN.AMP = False
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Perform evaluation after every N epochs (set to -1 to test after training)
_C.TEST.EVAL_STEP = 200
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 1
# Perform evaluation only
_C.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '0, 1'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = '/data/CUI/briar/data_working/krishna.regmi/cal/logs/'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'res50-ce-cal'


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# Number of frames in a tracklet
_C.DATALOADER.SEQ_LEN = 8
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# Which gallery set for test, options: 1, 2
_C.TEST.GALLERY_ID = 1
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234
_C.SOLVER.MARGIN = 0.3

# stage1
# ---------------------------------------------------------------------------- #
# Name of optimizer
_C.SOLVER.STAGE1 = CN()

_C.SOLVER.STAGE1.IMS_PER_BATCH = 64

_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE1.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE1.BASE_LR = 3e-4
# Momentum
_C.SOLVER.STAGE1.MOMENTUM = 0.9

# Settings of weight decay
_C.SOLVER.STAGE1.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE1.WEIGHT_DECAY_BIAS = 0.0005

# warm up factor
_C.SOLVER.STAGE1.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE1.LR_MIN = 0.000016

_C.SOLVER.STAGE1.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE1.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE1.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE1.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE1.LOG_PERIOD = 100
# epoch number of validation
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
# _C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.EVAL_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Solver
# stage1
# ---------------------------------------------------------------------------- #
_C.SOLVER.STAGE2 = CN()

_C.SOLVER.STAGE2.IMS_PER_BATCH = 64
# Name of optimizer
_C.SOLVER.STAGE2.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE2.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE2.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.STAGE2.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.STAGE2.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.STAGE2.MOMENTUM = 0.9
# Margin of triplet loss
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.STAGE2.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.STAGE2.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.STAGE2.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE2.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.STAGE2.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STAGE2.STEPS = (40, 70)
# warm up factor
_C.SOLVER.STAGE2.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE2.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE2.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE2.LR_MIN = 0.000016


_C.SOLVER.STAGE2.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE2.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE2.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE2.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE2.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE2.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.STAGE2.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
# Path where inference results get saved
_C.EVAL_DIR = ""


def update_config(config, args):
    config.defrost()
    config.merge_from_file(args.cfg)

    # merge from specific arguments
    if args.root:
        config.DATA.ROOT = args.root
    if args.output:
        config.OUTPUT = args.output

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.eval:
        config.EVAL_MODE = True
    
    if args.tag:
        config.TAG = args.tag

    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.gpu:
        config.GPU = args.gpu
    if args.amp:
        config.TRAIN.AMP = True

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG)

    config.freeze()


def get_vid_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config
