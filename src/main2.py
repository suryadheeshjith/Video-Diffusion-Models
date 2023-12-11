import argparse
import copy
import datetime
import glob
import logging
import numpy as np
import os
import shutil
import sys
import torch
import traceback
import time
import yaml
import random
# import torch.utils.tensorboard as tb
# from hanging_threads import start_monitoring
# start_monitoring(seconds_frozen=10, test_interval=100)

from runners import *

def override_config(args):
    args.command = 'python ' + ' '.join(sys.argv)
    args.exp = args.output_dir

    if args.data.dataset.upper() == "IMAGENET":
        if args.data.classes is None:
            args.data.classes = []
        elif args.data.classes == 'dogs':
            args.data.classes = list(range(151, 269))
        assert isinstance(args.data.classes, list), "args.data.classes must be a list!"
    args.sampling.subsample = args.subsample or args.sampling.subsample
    args.fast_fid.batch_size = args.fid_batch_size or args.fast_fid.batch_size
    args.fast_fid.num_samples = args.fid_num_samples or args.fast_fid.num_samples
    args.fast_fid.pr_nn_k = args.pr_nn_k or args.fast_fid.pr_nn_k
    if args.no_ema:
        args.model.ema = False

    if args.sampling.fvd and args.sampling.num_frames_pred < 10:
        print(" <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< WARNING: Cannot test FVD when sampling.num_frames_pred < 10 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        args.sampling.fvd =  False

    if args.model.output_all_frames: 
        noise_in_cond = True # if False, then wed predict the input-cond frames z, but the z is zero everywhere which is weird and seems irrelevant to predict. So we stick to the noise_in_cond case.

    assert not args.model.cond_emb or (args.model.cond_emb and args.data.prob_mask_cond > 0)

    if args.data.prob_mask_sync:
        assert args.data.prob_mask_cond > 0 and args.data.prob_mask_cond == args.data.prob_mask_future

    # Below were all commented!!

    if args.sampling.fvd and args.data.channels != 3:
       print(" <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< WARNING: Cannot test FVD when image is greyscale >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
       args.sampling.fvd = False
    
    if args.sampling.preds_per_test > 1:
        assert args.sampling.preds_per_test >= 5, f"preds_per_test can be either 1, or >=5. Got {args.sampling.preds_per_test}"

    # Override if interpolation
    # if args.data.num_frames_future > 0:
    #     args.sampling.num_frames_pred = args.data.num_frames
    
    return args


def setup_logger(args):
    if args.mode == "train":
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def setup_sampling(args):
    if args.ckpt is not None:
        args.sampling.ckpt_id = args.ckpt
    if args.sampling.ckpt_id == 0 :
        args.sampling.ckpt_id = None
    if args.final_only is not None:
        args.sampling.final_only = args.final_only

    if args.sampling.final_only:
        os.makedirs(os.path.join(args.output_dir, 'image_samples'), exist_ok=True)
        args.image_folder = os.path.join(args.output_dir, 'image_samples', args.image_folder)
    else:
        os.makedirs(os.path.join(args.output_dir, f'image_samples_{args.sampling.ckpt_id}'), exist_ok=True)
        args.image_folder = os.path.join(args.output_dir, f'image_samples_{args.sampling.ckpt_id}', args.image_folder)

    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(f"Image folder {args.image_folder} already exists.\nOverwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    
    return args


def setup_video_gen(args):
    args.sampling.ckpt_id = args.ckpt or args.sampling.ckpt_id
    args.final_only = True

    os.makedirs(os.path.join(args.output_dir, 'video_samples'), exist_ok=True)
    args.video_folder = os.path.join(args.output_dir, 'video_samples', args.video_folder)


    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input(f"Video folder {args.video_folder} already exists.\nOverwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(args.video_folder)
            os.makedirs(args.video_folder)
        else:
            print("Output video folder exists. Program halted.")
            sys.exit(0)


    return args


def setup_fast_fid(args):
    args.fast_fid.begin_ckpt = args.ckpt or args.fast_fid.begin_ckpt
    args.fast_fid.end_ckpt = args.end_ckpt or args.fast_fid.end_ckpt
    args.fast_fid.freq = args.freq or getattr(args.fast_fid, "freq", 5000)

    os.makedirs(os.path.join(args.output_dir, 'fid_samples'), exist_ok=True)
    args.image_folder = os.path.join(args.output_dir, 'fid_samples', args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = False
        else:
            response = input(f"Image folder {args.image_folder} already exists.\n "
                                "Type Y to delete and start from an empty folder?\n"
                                "Type N to overwrite existing folders (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)

    
    return args

def setup_test(args):
    args.test.begin_ckpt = args.ckpt or args.test.begin_ckpt
    args.test.end_ckpt = args.end_ckpt or args.test.end_ckpt
    args.test.freq = args.freq or getattr(args.test, "freq", 5000)
    
    return args


def main(args):
    args = override_config(args)
    # setup_logger(args)

    logging.info("Config= {}".format(args))
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    device = torch.device(args.device)
    fix_random_seeds(args.seed)
    torch.backends.cudnn.benchmark = True

    try:
        runner = NCSNRunner(args, args, args) # Remember, all are the same for me

        if args.mode == "test":
            args = setup_test(args)
            runner.test()
        
        elif args.mode == "sample":
            args = setup_sampling(args)
            runner.sample()
        
        elif args.mode == "video_gen":
            args = setup_video_gen(args)
            runner.video_gen()

        elif args.mode == "fast_fid":
            args = setup_fast_fid(args)
            runner.fast_fid()

        elif args.interact:
            pass

        else:
            runner.train()
    except:
        logging.error(traceback.format_exc())

    logging.info(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    if not args.interact:
        sys.exit()