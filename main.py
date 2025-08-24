import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import logging
import time
import sys
import yaml
import datetime
import argparse
from scipy.special import rel_entr
from model import PQC, Discriminator
from dataset import get_real_batch, get_bars_and_stripes
from utils import numpy_to_gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", required=True)
    args = parser.parse_args()
    output_dir = args.o
    with open(f"{output_dir}/param.yaml", "r") as f:
        param = yaml.safe_load(f)
    with open(f"config.yaml", "r") as f:
        config = yaml.safe_load(f)
  
    length = param["length"]
    size = length**2
    n_layers = param["n_layers"]
    bs_dis = param["bs_dis"]
    bs_gen = param["bs_gen"]
    lr_dis = param["lr_dis"]
    lr_gen = param["lr_gen"]
    features = param["features"]
    d_steps = param["d_steps"]
    g_steps = param["g_steps"]
    epochs = config["epochs"]
    save_epoch = config["save_epoch"]
    log_epoch = config["log_epoch"]
    epsilon = config["epsilon"]
    logfile = config["logfile"]
    accuracy_file = config["accuracy_file"]
    loss_dis_file = config["loss_dis_file"]
    loss_gen_file = config["loss_gen_file"]
    kl_div_file = config["kl_div_file"]
    theta_val_file = config["theta_val_file"]
    device = torch.device("cuda")
    is_tty = sys.stdout.isatty()
  
    on_mps = param.get("on_mps", False)
    if on_mps:
        v_mps = param["v_mps"]
        pqc = PQC(n_layers, size, bs_dis, bs_gen, lr_gen, device, on_mps, v_mps)
    else:
        pqc = PQC(n_layers, size, bs_dis, bs_gen, lr_gen, device)
    discriminator = Discriminator(size, features).to(device)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=lr_dis)
    bce_logits = nn.BCEWithLogitsLoss()
    target_list = get_bars_and_stripes(length)
    target_set = set(tuple(target) for target in target_list)
  
    loss_dis_list = []
    loss_gen_list = []
    accuracy_list = []
    kl_div_list = []
    theta_val_list = []
  
    logging.basicConfig(
        filename=f"{output_dir}/{logfile}",
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    lg = logging.getLogger("qiskit")
    lg.setLevel(logging.WARNING)
    lg.propagate = False
  
    with open(f"{output_dir}/{logfile}", "a") as f:
        f.write("===========================\n")
  
    torch.cuda.synchronize()
    start_time = time.time()
    logging.info("Learning started.")
    
    for epoch in tqdm(range(1, epochs + 1), desc="running", disable=not is_tty):
        for _ in range(d_steps):
            discriminator.train()
            x_real = numpy_to_gpu(get_real_batch(length, bs_dis))
            x_fake = numpy_to_gpu(pqc.run())
            y_real = discriminator(x_real)
            y_fake = discriminator(x_fake)
            loss_dis = bce_logits(y_real, torch.ones_like(y_real)) \
                       + bce_logits(y_fake, torch.zeros_like(y_fake))
            optimizer_dis.zero_grad()
            loss_dis.backward()
            optimizer_dis.step()
      
        for _ in range(g_steps):
            pqc.step(discriminator)
      
        if epoch % save_epoch == 0:
            predict_list = pqc.run(mode="G")
            count = sum(tuple(predict) in target_set for predict in predict_list)
            accuracy = count / len(predict_list)
            accuracy_list.append(accuracy)
      
            x_fake = numpy_to_gpu(predict_list)
            y_fake = discriminator(x_fake).detach()
            loss_gen = bce_logits(y_fake, torch.ones_like(y_fake))
            loss_dis_list.append(loss_dis.item())
            loss_gen_list.append(loss_gen.item())
      
            selected_list = np.zeros(len(target_list))
            for predict in predict_list:
                for i, target in enumerate(target_list):
                    if np.array_equal(predict, target):
                        selected_list[i] += 1
                        break
            selected_list += epsilon
            p_fake = selected_list / len(predict_list)
            p_real = np.ones(len(target_list)) / len(target_list)
            kl_div = np.sum(rel_entr(p_real, p_fake))
            kl_div_list.append(kl_div)
  
            theta_val_list.append(pqc.theta_val.copy())
  
        if epoch % log_epoch == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            formatted_time = str(datetime.timedelta(seconds=int(elapsed)))
            logging.info(
                f"Learning in progrees: {epoch}/{epochs} epochs completed. "
                f"Elapsed time: {formatted_time}."
            )
  
            np.save(f"{output_dir}/{accuracy_file}", np.array(accuracy_list))
            np.save(f"{output_dir}/{loss_dis_file}", np.array(loss_dis_list))
            np.save(f"{output_dir}/{loss_gen_file}", np.array(loss_gen_list))
            np.save(f"{output_dir}/{kl_div_file}", np.array(kl_div_list))
            np.save(f"{output_dir}/{theta_val_file}", np.array(theta_val_list))
  
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed = end_time - start_time
    formatted_time = str(datetime.timedelta(seconds=int(elapsed)))
    logging.info(f"Learning completed. Total time: {formatted_time}.")
  
    with open(f"{output_dir}/{logfile}", "a") as f:
        f.write("===========================\n")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("GPU: NOT FOUND - exiting")
    else:
        print("GPU: FOUND - starting")
        main()
