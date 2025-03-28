import random
from os.path import join

import pandas as pd
import torch
import torchaudio

from Checkpoints.UniFxModel import uniamp_confs
from Utils import uniamp_models


class RandomUniFxEffect(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        check_dir = join("Checkpoints", "UniFxModel")
        conf_num = 102
        model_name = "GiddyDisco113"
        ep = 3

        if seed is not None:
            random.seed(seed)

        model_dir = join(check_dir, f"conf{conf_num}-UniFx-{model_name}-ep{ep}.pt")
        config = uniamp_confs.main(model_conf=conf_num)

        self.df = pd.read_csv(join(check_dir, "train_index.csv"), index_col=0)
        self.num_devices = self.df.shape[0]

        state_dict = torch.load(
            model_dir, map_location=torch.device("cpu"), weights_only=True
        )
        self.model = uniamp_models.TCN(**config, num_emb=self.num_devices)
        self.model.load_state_dict(state_dict)

        self.counter = 0

    def forward(self, x):
        device_ids = [3, 8, 10, 14, 19]
        device_id = device_ids[self.counter % len(device_ids)]
        self.counter += 1

        device_id = random.choice(range(self.num_devices))

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
            f"uniamp_out_{i+1}_nn.wav", output[0, :, :].detach().cpu(), sample_rate=fs
        )
        print(f"Output {i+1} saved as 'uniamp_out_{i+1}_nn.wav'")


if __name__ == "__main__":
    main()
