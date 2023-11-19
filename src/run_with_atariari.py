from atariari.benchmark.episodes import get_episodes

def main(args):
    tr_episodes, val_episodes = get_episodes(env_name=args.env_name,
                 steps=args.steps,
                 seed=args.seed,
                 collect_mode=args.collect_mode,
                 train_mode="train_encoder",
                 min_episode_length=args.min_episode_length) 

                                        
    print("Num train episodes: ", len(tr_episodes))
    print("Num val episodes: ", len(val_episodes))