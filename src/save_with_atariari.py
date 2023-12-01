from atariari.benchmark.episodes import get_episodes

from functools import partial 
from pathlib import Path

import hydra
from hydra.utils import instantiate
import torch
import omegaconf
import numpy as np

from agent import Agent
from envs import SingleProcessEnv, WorldModelEnv
from game import AgentEnv, EpisodeReplayEnv, Save
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
import src_utils.misc as misc

import json
from einops import rearrange

def get_pretrained_iris_encoder(args, device):
    env_fn = partial(instantiate, config=args.iris.env.test)
    test_env = SingleProcessEnv(env_fn)

    h, w, _ = test_env.env.unwrapped.observation_space.shape
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]
    
    tokenizer = instantiate(args.iris.tokenizer)
    world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions, config=instantiate(args.iris.world_model))
    actor_critic = ActorCritic(**args.iris.actor_critic, act_vocab_size=test_env.num_actions)
    agent = Agent(tokenizer, world_model, actor_critic).to(device)
    agent.load(Path(args.checkpoint_dir) / "{}.pt".format(args.checkpoint_file), device)    

    tokenizer = AgentEnv(agent, test_env, args.iris.env.keymap, do_reconstruction=False).agent.tokenizer
    return tokenizer


def main(args):
    resolved_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    print("{}".format(resolved_args).replace(', ', ',\n'))
    
    # Setup dataset
    tr_episodes, val_episodes = get_episodes(env_name=args.env_name,
                 steps=args.steps,
                 seed=args.seed,
                 color=args.color,
                 collect_mode=args.collect_mode,
                 train_mode="train_encoder",
                 min_episode_length=args.min_episode_length) 

                                        
    print("Num train episodes: ", len(tr_episodes))
    print("Num val episodes: ", len(val_episodes))
    print("Frame shape: ", tr_episodes[0][0].shape)

    device = torch.device(args.device)

    # Get pretrained encoder
    if args.encoder == "iris":
        encoder = get_pretrained_iris_encoder(args, device)
    else:
        print("Enter valid encoder")
        return

    print("Loaded encoder")
    
    for i, tr_one_episode in enumerate(tr_episodes):
        
        # change this. Need to use dataset and dataloader for identical batch sizes
        ep_length = len(tr_one_episode)
        tr_one_episode = torch.cat(tr_one_episode)
        tr_one_episode = rearrange(tr_one_episode, '(t c) h w -> t c h w', t=ep_length, c=3 if args.color else 1) 

        tr_one_episode = tr_one_episode.to(device, non_blocking=True)

        episode_buffer = []
        encoder_buffer = []
        for frame in tr_one_episode:
            encoded_frame = encoder.encode(frame, should_preprocess=True).z.detach().cpu()

            episode_buffer.append(np.array(frame.cpu()))
            encoder_buffer.append(np.array(encoded_frame))

        misc.save_recording(Path(args.save_dir) / args.env_name / "train", str(i+1), np.stack(episode_buffer), np.stack(encoder_buffer))


    for i, tr_one_episode in enumerate(val_episodes):
        
        # change this. Need to use dataset and dataloader for identical batch sizes
        ep_length = len(tr_one_episode)
        tr_one_episode = torch.cat(tr_one_episode)
        tr_one_episode = rearrange(tr_one_episode, '(t c) h w -> t c h w', t=ep_length, c=3 if args.color else 1) 

        tr_one_episode = tr_one_episode.to(device, non_blocking=True)

        episode_buffer = []
        encoder_buffer = []
        for frame in tr_one_episode:
            encoded_frame = encoder.encode(frame, should_preprocess=True).z.detach().cpu()

            episode_buffer.append(np.array(frame.cpu()))
            encoder_buffer.append(np.array(encoded_frame))

        misc.save_recording(Path(args.save_dir) / args.env_name / "val", str(i+1), np.stack(episode_buffer), np.stack(encoder_buffer))




