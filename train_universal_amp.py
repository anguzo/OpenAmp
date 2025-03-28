import argparse
import os
from os.path import join

import auraloss
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Open_Amp import data_manager
from Utils import losses
from Utils import uniamp_configs as configs
from Utils import uniamp_dataloader, uniamp_models


def train_one_epoch(epoch_index, writer):
    for i, batch in tqdm(enumerate(train_dloader), total=len(train_dloader)):

        inputs, targets, enc = batch

        optimizer.zero_grad()

        outputs = model(inputs.to(device), enc.to(device))
        outputs, targets = outputs[:, :, train_dset.lead_in_samps :], targets[
            :, :, train_dset.lead_in_samps :
        ].to(device)

        # Compute the loss and its gradients
        td_l = esr(output=outputs, target=targets)
        fd_l = mrsl(x=outputs, y=targets)
        loss = td_l + fd_l
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        writer.add_scalar(
            "train/loss_total", loss.item(), epoch_index * len(train_dloader) + i
        )
        writer.add_scalar(
            "train/loss_esr", td_l.item(), epoch_index * len(train_dloader) + i
        )
        writer.add_scalar(
            "train/aura_mrstft", fd_l.item(), epoch_index * len(train_dloader) + i
        )


def val_loop(network, dloader, current_epoch, writer):
    amp_names = []
    esr_collected = []
    mrsl_collected = []
    metrics = {}

    with torch.inference_mode():
        for i, batch in tqdm(enumerate(dloader), total=len(dloader)):
            for b_idx, b in enumerate(batch):
                batch[b_idx] = b.to(device)
            inputs, targets, enc = [item[0] for item in batch]
            amp_idx = torch.argmax(enc, dim=-1)
            assert len(torch.unique(amp_idx)) == 1  # make sure batch is all same amp
            amp_name = dloader.dataset.get_model_name(amp_idx[0])

            # process audio
            outputs = network(inputs, enc)
            outputs, targets = (
                outputs[:, :, val_dset.lead_in_samps :],
                targets[:, :, val_dset.lead_in_samps :],
            )

            if i % (len(dloader) // 3) == 0:
                for m in range(outputs.shape[0]):
                    if m in [0, 2, 4, 6]:
                        writer.add_audio(
                            f"model_{amp_name}_clip_{m}",
                            outputs[m, 0, :].cpu(),
                            current_epoch,
                            sample_rate=44100,
                        )
                        if current_epoch == 0:
                            writer.add_audio(
                                f"model_{amp_name}_clip_{m}_target",
                                targets[m, 0, :].cpu(),
                                current_epoch,
                                sample_rate=44100,
                            )

            esr_loss = esr(output=outputs, target=targets)
            spec_loss = mrsl(x=outputs, y=targets)

            esr_collected.append(esr_loss)
            mrsl_collected.append(spec_loss)
            amp_names.append(amp_name)

    esr_collected = torch.tensor(esr_collected)
    mrsl_collected = torch.tensor(mrsl_collected)
    metrics["valid/loss_esr_mean"] = esr_collected.mean().item()
    metrics["valid/loss_esr_min"] = esr_collected.min().item()
    metrics["valid/loss_esr_max"] = esr_collected.max().item()
    metrics["valid/loss_mrsl_mean"] = mrsl_collected.mean().item()
    metrics["valid/loss_mrsl_min"] = mrsl_collected.min().item()
    metrics["valid/loss_mrsl_max"] = mrsl_collected.max().item()
    for i, amp_name in enumerate(amp_names):
        metrics[f"valid/AmpSpecificLoss_esr/{amp_name}"] = esr_collected[i].item()
        metrics[f"valid/AmpSpecificLoss_mrsl/{amp_name}"] = mrsl_collected[i].item()

    metrics["epoch"] = current_epoch
    for key, value in metrics.items():
        writer.add_scalar(key, value, current_epoch)
    print(f"epoch {current_epoch} val loss esr: {esr_collected.mean().item()}")
    return metrics


parser = argparse.ArgumentParser()
parser.add_argument(
    "-tc", "--train_config", type=int, default=1, help="config for model training"
)
parser.add_argument("-nw", "--num_workers", type=int, default=0)
parser.add_argument("-tbs", "--train_batch_size", type=int, default=0)
parser.add_argument("-vbs", "--val_batch_size", type=int, default=0)

args = parser.parse_args()

if __name__ == "__main__":
    print(f"cpu cores - {os.cpu_count()}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("using cuda")
        conf = configs.main(config_set=args.train_config)
        num_workers = 24
    else:
        device = torch.device("cpu")
        print("using cpu")
        conf = configs.main(config_set=0)
        num_workers = 8
        args.train_config = 0

    if args.num_workers > 0:
        num_workers = args.num_workers

    save_dir = join("Results", f"conf_{args.train_config}")
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=save_dir)

    np.random.seed(42)
    device_list = np.random.choice(
        list(conf["devices"]), len(conf["devices"]), replace=False
    )

    train_devs = data_manager.get_dev_table(
        save_loc=save_dir,
        model_list=device_list[0:-10],
        save_name="train_index",
        cnd_res=conf["cond_res"],
    )
    test_devs = data_manager.get_dev_table(
        save_loc=save_dir,
        model_list=device_list[-10:],
        save_name="test_index",
        cnd_res=conf["cond_res"],
    )

    data_manager.init_dataset(devices=train_devs, **conf["val_data"])

    train_dset = uniamp_dataloader.OnlineGenerationDataset(
        devices=train_devs, **conf["train_data"]
    )
    val_dset = uniamp_dataloader.LoadingDataset(
        devices=train_devs,
        dataset_loc=conf["val_data"]["dataset_name"],
        seg_len=conf["val_data"]["seg_len"],
        lead_in=conf["val_data"]["lead_in"],
    )

    train_dloader = DataLoader(
        train_dset,
        batch_size=conf["t_bs"],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_dloader = DataLoader(
        val_dset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    model = uniamp_models.TCN(**conf["TCN"], num_emb=train_dset.num_models).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mrsl = auraloss.freq.MultiResolutionSTFTLoss()
    esr = losses.ESRLoss()

    validation_sanity = False

    start_epoch = 0
    model_path = None
    for file in os.listdir(save_dir):
        if file.startswith("model_weights_ep-") and file.endswith(".pt"):
            epoch_num = int(file.split("-")[-1].split(".")[0])
            if epoch_num > start_epoch:
                start_epoch = epoch_num
                model_path = join(save_dir, file)

    if model_path:
        model.load_state_dict(torch.load(model_path))
        start_epoch += 1

    for n in range(start_epoch, 1000):
        if n > start_epoch or validation_sanity:
            print(f"starting validation epoch {n}")
            val_loop(model, val_dloader, current_epoch=n, writer=writer)
        print(f"starting training epoch {n}")
        train_one_epoch(n, writer=writer)

        torch.save(model.state_dict(), join(save_dir, f"model_weights_ep-{n}.pt"))

    val_loop(model, val_dloader, current_epoch=n + 1, writer=writer)
    writer.close()
