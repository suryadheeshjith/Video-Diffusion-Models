import numpy as np
from pathlib import Path

def save_recording(record_dir: Path, ep_num, frames, encodings):
    record_dir.mkdir(exist_ok=True, parents=True)
    
    frames_file = ep_num + '_frames.npy'
    encodings_file = ep_num + '_encodings.npy'

    np.save(record_dir / frames_file, frames)
    np.save(record_dir / encodings_file, encodings)
    print("Saved recording ", ep_num)