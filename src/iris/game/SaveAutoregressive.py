from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

from einops import rearrange
import torch
import gym
import numpy as np
import pygame
from PIL import Image
import torchvision.transforms.functional as TF
from torch.distributions.categorical import Categorical

from models.actor_critic import ActorCritic
from envs import WorldModelEnv
from game.keymap import get_keymap_and_action_names
from utils import make_video


class SaveAutoregressive:
    def __init__(self, env: Union[gym.Env, WorldModelEnv], actor_critic: ActorCritic, keymap_name: str, size: Tuple[int, int], fps: int, verbose: bool, record_mode: bool, save_dir: Path, episode_steps: int) -> None:
        self.env = env
        # self.height, self.width = size
        self.actor_critic = actor_critic
        self.size = size
        self.fps = fps
        self.verbose = verbose
        self.record_mode = True
        self.keymap, self.action_names = get_keymap_and_action_names(keymap_name)
        self.record_dir = save_dir
        self.counter = 0
        self.episode_steps = episode_steps
        self.current_steps = 0

        # print('Actions:')
        # for key, idx in self.keymap.items():
        #     print(f'{pygame.key.name(key)}: {self.action_names[idx]}')

    def get_next_token(self, action_token):
        action_token = action_token.clone().detach() if isinstance(action_token, torch.Tensor) else torch.tensor(action_token, dtype=torch.long)
        next_token = self.env.agent.world_model(action_token, should_predict_next_obs=True)
        
        return next_token
    
    def get_action_token(self, obs):
        outputs_ac = self.actor_critic(obs)
        action_token = Categorical(logits=outputs_ac.logits_actions).sample()   

        return action_token
    
    def atari_transform(self, img, size):
        return TF.resize(img, (size,size))

    
    def run(self) -> None:

        if isinstance(self.env, gym.Env):
            _, info = self.env.reset(return_info=True)
            img = info['rgb']
        else:
            self.actor_critic.reset(n=1)
            img = self.env.reset()
            # print("Img shape: ",img.shape)

        episode_buffer = []
        token_buffer = []
        segment_buffer = []
        recording = False

        do_reset, do_wait = False, False
        done = False
        action_token = self.get_action_token(img)

        while self.current_steps < self.episode_steps:

            obs, _, done, _ = self.env.step(action_token, should_predict_next_obs=True)
            # print("Obs shape: ",obs.shape)
            action_token = self.get_action_token(obs)
            state_token = self.env.get_tokens()
            # print("token shape: ",state_token.shape)
            self.current_steps += 1

            if self.record_mode:
                world_model_frame = self.env.render()
                episode_buffer.append(np.array(world_model_frame))
                token_buffer.append(state_token.detach().cpu().numpy())

            if do_reset or done:
                self.actor_critic.reset(n=1)
                img = self.env.reset_from_initial_observations(img)
                action_token = self.get_action_token(img)
                do_reset = False

                if self.record_mode:
                    print("episode length: ",len(episode_buffer))
                    self.save_recording(np.stack(episode_buffer), np.stack(token_buffer))
                    episode_buffer = []
                    token_buffer = []

    def save_recording(self, frames, tokens):
        self.record_dir.mkdir(exist_ok=True, parents=True)
        self.counter += 1
        frames_file = str(self.counter) + '_frames.npy'
        tokens_file = str(self.counter) + '_tokens.npy'

        np.save(self.record_dir / frames_file, frames)
        np.save(self.record_dir / tokens_file, tokens)
        print("Saved recording ", self.counter)