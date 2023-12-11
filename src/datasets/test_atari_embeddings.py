from atari import AtariDataset, get_atari_transform
import torch 
from torch.utils.data import DataLoader
import argparse

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def conditioning_fn(config, X, num_frames_pred=0, prob_mask_cond=0.0, prob_mask_future=0.0, conditional=True, encodings=None):
    imsize = config.data.image_size
    if not conditional:
        return X.reshape(len(X), -1, imsize, imsize), None, None

    cond = config.data.num_frames_cond
    train = config.data.num_frames
    pred = num_frames_pred
    future = getattr(config.data, "num_frames_future", 0)

    # Frames to train on / sample
    pred_frames = X[:, cond:cond+pred].reshape(len(X), -1, imsize, imsize)

    # Condition (Past)
    cond_frames = X[:, :cond].reshape(len(X), -1, imsize, imsize)

    if prob_mask_cond > 0.0:
        cond_mask = (torch.rand(X.shape[0], device=X.device) > prob_mask_cond)
        cond_frames = cond_mask.reshape(-1, 1, 1, 1) * cond_frames
        cond_mask = cond_mask.to(torch.int32) # make 0,1
    else:
        cond_mask = None

    # Future
    if future > 0:

        if prob_mask_future == 1.0:
            future_frames = torch.zeros(len(X), config.data.channels*future, imsize, imsize)
            # future_mask = torch.zeros(len(X), 1, 1, 1).to(torch.int32) # make 0,1
        else:
            future_frames = X[:, cond+train:cond+train+future].reshape(len(X), -1, imsize, imsize)
            if prob_mask_future > 0.0:
                if getattr(config.data, "prob_mask_sync", False):
                    future_mask = cond_mask
                else:
                    future_mask = (torch.rand(X.shape[0], device=X.device) > prob_mask_future)
                future_frames = future_mask.reshape(-1, 1, 1, 1) * future_frames
            #     future_mask = future_mask.to(torch.int32) # make 0,1
            # else:
            #     future_mask = None

        cond_frames = torch.cat([cond_frames, future_frames], dim=1)
    
    if encodings is not None:
        pred_encodings = encodings[:, cond:cond+pred].reshape(len(X), -1, imsize, imsize)
        cond_frames = torch.cat([cond_frames, pred_encodings], dim=1)

    return pred_frames, cond_frames, cond_mask

def convert_encodings(encodings, enc_type, frame_shape):
    enc_shape = encodings.shape
    encodings = encodings.reshape(enc_shape[0]*enc_shape[1], *enc_shape[2:])

    if enc_type == "avg":
        encodings = torch.nn.functional.adaptive_avg_pool2d(encodings, (4, 2))
    elif enc_type == "max":
        encodings = torch.nn.functional.adaptive_max_pool2d(encodings, (4, 2))
    
    encodings = encodings.reshape(enc_shape[0], enc_shape[1], -1, *frame_shape[3:])
    return encodings

val_dataset = AtariDataset("/vast/sd5313/data/BreakoutNoFrameskip-v4/val", 9, get_atari_transform(64), 3)
print("Val Dataset Length with ep 3", len(val_dataset))
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)


for i, (frames, encodings) in enumerate(val_dataloader):
    print(frames.shape, encodings.shape)
    encodings = convert_encodings(encodings, "avg", frames.shape)

    # Define dummy config
    config = {
        "data": {
            "image_size": 64,
            "channels": 3,
            "num_frames": 5,
            "num_frames_cond": 2,
            "num_frames_future": 2,
            "prob_mask_cond": 0.5,
            "prob_mask_future": 0.5,
            "prob_mask_sync": False,
        }
    }
    config = dict2namespace(config)
    
    pred_frames, cond_frames, cond_mask = conditioning_fn(config, frames, num_frames_pred=config.data.num_frames, prob_mask_cond=config.data.prob_mask_cond, \
                    prob_mask_future=config.data.prob_mask_future, conditional=True, encodings=encodings)
    
    print(pred_frames.shape, cond_frames.shape, cond_mask.shape)


    # Flatten and reshape
    break