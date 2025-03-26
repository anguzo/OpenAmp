import random
from os.path import join

import pandas as pd
import torch
import torchaudio

from Utils import uniamp_models


class RandomUniFxEffect(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        check_dir = join("Results", "conf_1")

        if seed is not None:
            random.seed(seed)

        model_dir = join(check_dir, f"model_weights_ep-6.pt")
        config = {
            "layer_class": uniamp_models.GatedTCNLayerFilmConditioned,
            "channels": 16,
            "blocks": 2,
            "layers": 8,
            "dilation_growth": 2,
            "kernel_size": 3,
            "cond_pars": 64,
            "emb_dim": 64,
        }

        self.df = pd.read_csv(join(check_dir, "train_index.csv"), index_col=0)
        self.num_devices = self.df.shape[0]

        self.device_ids = list(range(self.num_devices))
        random.shuffle(self.device_ids)
        self.current_idx = 0

        state_dict = torch.load(
            model_dir, map_location=torch.device("cpu"), weights_only=True
        )
        self.model = uniamp_models.TCN(**config, num_emb=self.num_devices)
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        device_id = self.device_ids[self.current_idx]

        self.current_idx = (self.current_idx + 1) % self.num_devices

        if self.current_idx == 0:
            random.shuffle(self.device_ids)

        print(f"Selected device id: {device_id} - {self.df.iloc[device_id].model}")
        return self.model.proc_with_emb_id(x, device_id)


def main():
    random_model = RandomUniFxEffect()

    input_sig, fs = torchaudio.load("example_5.wav")
    if input_sig.ndim == 2:
        input_sig = input_sig.unsqueeze(0)

    for i in range(5):
        output = random_model(input_sig)
        torchaudio.save(
            f"uniamp_out_{i+1}_alt_balanced.wav",
            output[0, :, :].detach().cpu(),
            sample_rate=fs,
        )
        print(f"Output {i+1} saved as 'uniamp_out_{i+1}_balanced.wav'")


if __name__ == "__main__":
    main()
