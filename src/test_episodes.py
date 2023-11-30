from atariari.benchmark.episodes import get_episodes
tr_episodes, val_episodes =get_episodes(env_name="BreakoutNoFrameskip-v4",
                 steps=5000,
                 seed=0,
                 color=False,
                 collect_mode="pretrained_ppo_color",
                 train_mode="train_encoder",
                 min_episode_length=64)

print(len(tr_episodes),len(val_episodes))
