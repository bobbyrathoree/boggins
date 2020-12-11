import os.path as osp
from collections import OrderedDict
import json
import os
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import random
import torch
import logging


def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt["phase"]
    if phase == "train":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt["batch_size"],
            shuffle=dataset_opt["use_shuffle"],
            num_workers=dataset_opt["n_workers"],
            drop_last=True,
            pin_memory=True,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
        )


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        print("Path already exists. Rename it to [{:s}]".format(new_name))
        logger = logging.getLogger("base")
        logger.info("Path already exists. Rename it to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def save_img(img, img_path, mode="RGB"):
    cv2.imwrite(img_path, img)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def parse(opt_path, is_train=True):
    json_str = ""
    with open(opt_path, "r") as f:
        for line in f:
            line = line.split("//")[0] + "\n"
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt["is_train"] = is_train
    scale = opt["scale"]

    for phase, dataset in opt["datasets"].items():
        phase = phase.split("_")[0]
        dataset["phase"] = phase
        dataset["scale"] = scale
        is_lmdb = False
        if "dataroot_HR" in dataset and dataset["dataroot_HR"] is not None:
            dataset["dataroot_HR"] = os.path.expanduser(dataset["dataroot_HR"])
            if dataset["dataroot_HR"].endswith("lmdb"):
                is_lmdb = True
        if "dataroot_HR_bg" in dataset and dataset["dataroot_HR_bg"] is not None:
            dataset["dataroot_HR_bg"] = os.path.expanduser(dataset["dataroot_HR_bg"])
        if "dataroot_LR" in dataset and dataset["dataroot_LR"] is not None:
            dataset["dataroot_LR"] = os.path.expanduser(dataset["dataroot_LR"])
            if dataset["dataroot_LR"].endswith("lmdb"):
                is_lmdb = True
        dataset["data_type"] = "lmdb" if is_lmdb else "img"

        if (
            phase == "train"
            and "subset_file" in dataset
            and dataset["subset_file"] is not None
        ):
            dataset["subset_file"] = os.path.expanduser(dataset["subset_file"])

    for key, path in opt["path"].items():
        if path and key in opt["path"]:
            opt["path"][key] = os.path.expanduser(path)
    if is_train:
        experiments_root = os.path.join(opt["path"]["root"], "experiments", opt["name"])
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = os.path.join(experiments_root, "models")
        opt["path"]["training_state"] = os.path.join(experiments_root, "training_state")
        opt["path"]["log"] = experiments_root
        opt["path"]["val_images"] = os.path.join(experiments_root, "val_images")

        if "debug" in opt["name"]:
            opt["train"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 2
            opt["logger"]["save_checkpoint_freq"] = 8
            opt["train"]["lr_decay_iter"] = 10
    else:
        results_root = os.path.join(opt["path"]["root"], "results", opt["name"])
        opt["path"]["results_root"] = results_root
        opt["path"]["log"] = results_root

    opt["network_G"]["scale"] = scale

    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


def check_resume(opt):
    logger = logging.getLogger("base")
    if opt["path"]["resume_state"]:
        if opt["path"]["pretrain_model_G"] or opt["path"]["pretrain_model_D"]:
            logger.warning(
                "pretrain_model path will be ignored when resuming training."
            )

        state_idx = osp.basename(opt["path"]["resume_state"]).split(".")[0]
        opt["path"]["pretrain_model_G"] = osp.join(
            opt["path"]["models"], "{}_G.pth".format(state_idx)
        )
        logger.info("Set [pretrain_model_G] to " + opt["path"]["pretrain_model_G"])
        if "gan" in opt["model"]:
            opt["path"]["pretrain_model_D"] = osp.join(
                opt["path"]["models"], "{}_D.pth".format(state_idx)
            )
            logger.info("Set [pretrain_model_D] to " + opt["path"]["pretrain_model_D"])
