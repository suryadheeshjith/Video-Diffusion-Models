from functools import partial 
from pathlib import Path

import hydra
from hydra.utils import instantiate
import torch
import omegaconf

from agent import Agent
from envs import SingleProcessEnv, WorldModelEnv
from game import AgentEnv, EpisodeReplayEnv, SaveAutoregressive
from models.actor_critic import ActorCritic
from models.world_model import WorldModel

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import json
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"



def main(args):    
    resolved_args = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    print("{}".format(resolved_args).replace(', ', ',\n'))
    
    device = torch.device(args.iris.common.device)
    assert args.mode in ('agent_in_env', 'agent_in_world_model')

    env_fn = partial(instantiate, config=args.iris.env.test)
    test_env = SingleProcessEnv(env_fn)

    if args.mode.startswith('agent_in_'):
        h, w, _ = test_env.env.unwrapped.observation_space.shape
    else:
        h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]
    
    tokenizer = instantiate(args.iris.tokenizer)
    world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions, config=instantiate(args.iris.world_model))
    actor_critic = ActorCritic(**args.iris.actor_critic, act_vocab_size=test_env.num_actions)
    agent = Agent(tokenizer, world_model, actor_critic).to(device)
    agent.load(Path(args.checkpoint_dir) / "{}.pt".format(args.checkpoint_file), device)    

    # env = AgentEnv(agent, test_env, args.iris.env.keymap, do_reconstruction=args.reconstruction)
    env = WorldModelEnv(agent.tokenizer, agent.world_model, device, env_fn())
    keymap = 'empty'
    if args.reconstruction:
        size[1] *= 3

    save = SaveAutoregressive.SaveAutoregressive(env, actor_critic ,keymap_name=keymap, size=64, fps=args.fps, verbose=bool(args.header), record_mode=bool(args.save_mode), save_dir=Path(args.save_dir) / args.dataset_name, episode_steps=args.iris.env.test.max_episode_steps)
    save.run()


if __name__ == "__main__":
    main()
