import psutil
import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys
import yaml
import pickle

from collections import OrderedDict
import cv2
from pathlib import Path
import random
import shutil

from episode import Episode

#####
# Stat utils
#####
def count_training_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_proc_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3


def get_GPU_mem():
    try:
        num = torch.cuda.device_count()
        mem = 0
        for i in range(num):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            mem += (mem_total - mem_free)/1024**3
        return mem
    except:
        return 0


def convert_time_stamp_to_hrs(time_day_hr):
        time_day_hr = time_day_hr.split(",")
        if len(time_day_hr) > 1:
            days = time_day_hr[0].split(" ")[0]
            time_hr = time_day_hr[1]
        else:
            days = 0
            time_hr = time_day_hr[0]
        # Hr
        hrs = time_hr.split(":")
        return float(days)*24 + float(hrs[0]) + float(hrs[1])/60 + float(hrs[2])/3600


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99, save_seq=True):
        self.momentum = momentum
        self.save_seq = save_seq
        if self.save_seq:
            self.vals, self.steps = [], []
        self.reset()

    def reset(self):
        self.val, self.avg = None, 0

    def update(self, val, step=None):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if step is not None:
                self.steps.append(step)


#####
# Torch utils
#####

def stretch_image(X, ch, imsize):
    return X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 2, 1, 4, 3).reshape(len(X), ch, -1, imsize).permute(0, 1, 3, 2)

def convert_encodings(encodings, enc_type, frame_shape):
    enc_shape = encodings.shape
    encodings = encodings.reshape(enc_shape[0]*enc_shape[1], *enc_shape[2:])

    if enc_type == "avg":
        encodings = torch.nn.functional.adaptive_avg_pool2d(encodings, (4, 2))
    elif enc_type == "max":
        encodings = torch.nn.functional.adaptive_max_pool2d(encodings, (4, 2))
    elif enc_type == "bilinear":
        encodings = torch.nn.functional.interpolate(encodings, size=(6, 4), mode='bilinear', align_corners=False)
    
    encodings = encodings.reshape(enc_shape[0], enc_shape[1], -1, *frame_shape[3:])
    return encodings

#####
# Model utils
#####

def get_sigma_hooks(config):
    if config.training.log_all_sigmas:
        ### Commented out training time logging to save time.
        test_loss_per_sigma = [None for _ in range(getattr(config.model, 'num_classes'))]

        def hook(loss, labels):
            # for i in range(len(sigmas)):
            #     if torch.any(labels == i):
            #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
            pass

        def tb_hook():
            # for i in range(len(sigmas)):
            #     if test_loss_per_sigma[i] is not None:
            #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
            #                              global_step=step)
            pass

        def test_hook(loss, labels):
            for i in range(getattr(config.model, 'num_classes')):
                if torch.any(labels == i):
                    test_loss_per_sigma[i] = torch.mean(loss[labels == i])

        def test_tb_hook():
            for i in range(getattr(config.model, 'num_classes')):
                if test_loss_per_sigma[i] is not None:
                    tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                            global_step=step)

    else:
        hook = test_hook = None

        def tb_hook():
            pass

        def test_tb_hook():
            pass
        
    return hook, test_hook, tb_hook, test_tb_hook

#####
# Plot utils
#####

def savefig(path, bbox_inches='tight', pad_inches=0.1):
    try:
        plt.savefig(path, bbox_inches=bbox_inches, pad_inches=pad_inches)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        print(sys.exc_info()[0])


def plot_graphs(args, losses_train, losses_test, epochs, lr_meter, grad_norm, time_train, time_elapsed):
    # Losses
    plt.plot(losses_train.steps, losses_train.vals, label='Train')
    plt.plot(losses_test.steps, losses_test.vals, label='Test')
    plt.xlabel("Steps")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    plt.legend(loc='upper right')
    savefig(os.path.join(args.log_path, 'loss.png'))
    plt.yscale("log")
    savefig(os.path.join(args.log_path, 'loss_log.png'))
    plt.clf()
    plt.close()
    # Epochs
    plt.plot(losses_train.steps, epochs.vals)
    plt.xlabel("Steps")
    plt.ylabel("Epochs")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    savefig(os.path.join(args.log_path, 'epochs.png'))
    plt.clf()
    plt.close()
    # LR
    plt.plot(losses_train.steps, lr_meter.vals)
    plt.xlabel("Steps")
    plt.ylabel("LR")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    savefig(os.path.join(args.log_path, 'lr.png'))
    plt.clf()
    plt.close()
    # Grad Norm
    plt.plot(losses_train.steps, grad_norm.vals)
    plt.xlabel("Steps")
    plt.ylabel("Grad Norm")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    savefig(os.path.join(args.log_path, 'grad.png'))
    plt.yscale("log")
    savefig(os.path.join(args.log_path, 'grad_log.png'))
    plt.clf()
    plt.close()
    # Time train
    plt.plot(losses_train.steps, time_train.vals)
    plt.xlabel("Steps")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    savefig(os.path.join(args.log_path, 'time_train.png'))
    plt.clf()
    plt.close()
    # Time elapsed
    plt.plot(losses_train.steps[:len(time_elapsed.vals)], time_elapsed.vals)
    plt.xlabel("Steps")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    savefig(os.path.join(args.log_path, 'time_elapsed.png'))
    plt.clf()
    plt.close()

def plot_video_graphs_single(args, name, mses, psnrs, ssims, lpipss, fvds, calc_fvd,
                                best_mse, best_psnr, best_ssim, best_lpips, best_fvd):
    # MSE
    plt.plot(mses.steps, mses.vals)
    if best_mse['ckpt'] > -1:
        plt.scatter(best_mse['ckpt'], mses.vals[mses.steps.index(best_mse['ckpt'])], color='k')
        plt.text(best_mse['ckpt'], mses.vals[mses.steps.index(best_mse['ckpt'])], f"{mses.vals[mses.steps.index(best_mse['ckpt'])]:.04f}\n{best_mse['ckpt']}", c='r')
    plt.xlabel("Steps")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    # plt.legend(loc='upper right')
    savefig(os.path.join(args.log_path, f"mse_{name}.png"))
    plt.yscale("log")
    savefig(os.path.join(args.log_path, f"mse_{name}_log.png"))
    plt.clf()
    plt.close()
    # PSNR
    plt.plot(mses.steps, psnrs.vals)
    if best_psnr['ckpt'] > -1:
        plt.scatter(best_psnr['ckpt'], psnrs.vals[mses.steps.index(best_psnr['ckpt'])], color='k')
        plt.text(best_psnr['ckpt'], psnrs.vals[mses.steps.index(best_psnr['ckpt'])], f"{psnrs.vals[mses.steps.index(best_psnr['ckpt'])]:.04f}\n{best_psnr['ckpt']}", c='r')
    plt.xlabel("Steps")
    plt.ylabel("PSNR")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    # plt.legend(loc='upper right')
    savefig(os.path.join(args.log_path, f"psnr_{name}.png"))
    plt.yscale("log")
    savefig(os.path.join(args.log_path, f"psnr_{name}_log.png"))
    plt.clf()
    plt.close()
    # SSIM
    plt.plot(mses.steps, ssims.vals)
    if best_ssim['ckpt'] > -1:
        plt.scatter(best_ssim['ckpt'], ssims.vals[mses.steps.index(best_ssim['ckpt'])], color='k')
        plt.text(best_ssim['ckpt'], ssims.vals[mses.steps.index(best_ssim['ckpt'])], f"{ssims.vals[mses.steps.index(best_ssim['ckpt'])]:.04f}\n{best_ssim['ckpt']}", c='r')
    plt.xlabel("Steps")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    # plt.legend(loc='upper right')
    savefig(os.path.join(args.log_path, f"ssim_{name}.png"))
    plt.yscale("log")
    savefig(os.path.join(args.log_path, f"ssim_{name}_log.png"))
    plt.clf()
    plt.close()
    # LPIPS
    plt.plot(mses.steps, lpipss.vals)
    if best_lpips['ckpt'] > -1:
        plt.scatter(best_lpips['ckpt'], lpipss.vals[mses.steps.index(best_lpips['ckpt'])], color='k')
        plt.text(best_lpips['ckpt'], lpipss.vals[mses.steps.index(best_lpips['ckpt'])], f"{lpipss.vals[mses.steps.index(best_lpips['ckpt'])]:.04f}\n{best_lpips['ckpt']}", c='r')
    plt.xlabel("Steps")
    plt.ylabel("LPIPS")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    # plt.legend(loc='upper right')
    savefig(os.path.join(args.log_path, f"lpips_{name}.png"))
    plt.yscale("log")
    savefig(os.path.join(args.log_path, f"lpips_{name}_log.png"))
    plt.clf()
    plt.close()
    # FVD
    if calc_fvd:
        plt.plot(mses.steps, fvds.vals)
        if best_fvd['ckpt'] > -1:
            plt.scatter(best_fvd['ckpt'], fvds.vals[mses.steps.index(best_fvd['ckpt'])], color='k')
            plt.text(best_fvd['ckpt'], fvds.vals[mses.steps.index(best_fvd['ckpt'])], f"{fvds.vals[mses.steps.index(best_fvd['ckpt'])]:.04f}\n{best_fvd['ckpt']}", c='r')
        plt.xlabel("Steps")
        plt.ylabel("FVD")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        # plt.legend(loc='upper right')
        savefig(os.path.join(args.log_path, f"fvd_{name}.png"))
        plt.yscale("log")
        savefig(os.path.join(args.log_path, f"fvd_{name}_log.png"))
        plt.clf()
        plt.close()

def plot_fvd_gen(args, fvds3, best_fvd3):
    plt.plot(fvds3.steps, fvds3.vals)
    if best_fvd3['ckpt'] > -1:
        plt.scatter(best_fvd3['ckpt'], fvds3.vals[fvds3.steps.index(best_fvd3['ckpt'])], color='k')
        plt.text(best_fvd3['ckpt'], fvds3.vals[fvds3.steps.index(best_fvd3['ckpt'])], f"{fvds3.vals[fvds3.steps.index(best_fvd3['ckpt'])]:.04f}\n{best_fvd3['ckpt']}", c='r')
    plt.xlabel("Steps")
    plt.ylabel("FVD")
    plt.grid(True)
    plt.grid(visible=True, which='minor', axis='y', linestyle='--')
    # plt.legend(loc='upper right')
    savefig(os.path.join(args.log_path, 'fvd_gen.png'))
    plt.yscale("log")
    savefig(os.path.join(args.log_path, 'fvd_gen_log.png'))
    plt.clf()
    plt.close()


#####
# File utils
#####

def write_to_pickle(pickle_file, my_dict):
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as handle:
            old_dict = pickle.load(handle)
        for key in my_dict.keys():
            old_dict[key] = my_dict[key]
        my_dict = {}
        for key in sorted(old_dict.keys()):
            my_dict[key] = old_dict[key]
    with open(pickle_file, 'wb') as handle:
        pickle.dump(my_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def write_to_yaml(yaml_file, my_dict):
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            old_dict = yaml.load(f, Loader=yaml.FullLoader)
        for key in my_dict.keys():
            old_dict[key] = my_dict[key]
        my_dict = {}
        for key in sorted(old_dict.keys()):
            my_dict[key] = old_dict[key]
    with open(yaml_file, 'w') as f:
        yaml.dump(my_dict, f, default_flow_style=False)


#####
# IRIS utils
#####


def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class EpisodeDirManager:
    def __init__(self, episode_dir: Path, max_num_episodes: int) -> None:
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=False, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf')

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        ep_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith('episode_')]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(ep_paths, key=lambda ep_path: int(ep_path.stem.split('_')[1]))
            to_remove.unlink()
        episode.save(self.episode_dir / f'episode_{episode_id}_epoch_{epoch}.pt')

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            self.best_return = ep_return
            path_best_ep = [p for p in self.episode_dir.iterdir() if p.stem.startswith('best_')]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            episode.save(self.episode_dir / f'best_episode_{episode_id}_epoch_{epoch}.pt')


class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames):
    assert frames.ndim == 4 # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()

def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class EpisodeDirManager:
    def __init__(self, episode_dir: Path, max_num_episodes: int) -> None:
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=False, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf')

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        ep_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith('episode_')]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(ep_paths, key=lambda ep_path: int(ep_path.stem.split('_')[1]))
            to_remove.unlink()
        episode.save(self.episode_dir / f'episode_{episode_id}_epoch_{epoch}.pt')

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            self.best_return = ep_return
            path_best_ep = [p for p in self.episode_dir.iterdir() if p.stem.startswith('best_')]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            episode.save(self.episode_dir / f'best_episode_{episode_id}_epoch_{epoch}.pt')


class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames):
    assert frames.ndim == 4 # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()


def save_recording(record_dir: Path, ep_num, frames, encodings):
    record_dir.mkdir(exist_ok=True, parents=True)
    
    frames_file = ep_num + '_frames.npy'
    encodings_file = ep_num + '_encodings.npy'

    np.save(record_dir / frames_file, frames)
    np.save(record_dir / encodings_file, encodings)
    print("Saved recording ", ep_num)