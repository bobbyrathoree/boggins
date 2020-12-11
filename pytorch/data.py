import logging
import math
import os.path
import random
import numpy as np
import cv2
import lmdb
import pickle
import torch
import torch.utils.data as data


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), "{:s} is not a valid directory".format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, "{:s} has no valid image file".format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, "_keys_cache.p")
    logger = logging.getLogger("base")
    if os.path.isfile(keys_cache_file):
        logger.info("Read lmdb keys from cache: {}".format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            logger.info("Creating lmdb keys cache: {}".format(keys_cache_file))
            keys = [key.decode("ascii") for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, "wb"))
    paths = sorted([key for key in keys if not key.endswith(".meta")])
    return env, paths


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == "lmdb":
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == "img":
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError(
                "data_type [{:s}] is not recognized.".format(data_type)
            )
    return env, paths


def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode("ascii"))
        buf_meta = txn.get((path + ".meta").encode("ascii")).decode("ascii")
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(",")]
    img = img_flat.reshape(H, W, C)
    return img


def read_img(env, path):
    if env is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_lmdb_img(env, path)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def augment(img_list, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def channel_convert(in_c, tar_type, img_list):
    if in_c == 3 and tar_type == "gray":
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == "y":
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == "RGB":
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [
                [65.481, -37.797, 112.0],
                [128.553, -74.203, -93.786],
                [24.966, 112.0, -18.214],
            ],
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [
                [24.966, 112.0, -18.214],
                [128.553, -74.203, -93.786],
                [65.481, -37.797, 112.0],
            ],
        ) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0
    rlt = np.matmul(
        img,
        [
            [0.00456621, 0.00456621, 0.00456621],
            [0, -0.00153632, 0.00791071],
            [0.00625893, -0.00318811, 0],
        ],
    ) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[: H - H_r, : W - W_r, :]
    else:
        raise ValueError("Wrong img ndim: [{:d}].".format(img.ndim))
    return img


def cubic(x):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * (((absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(
    in_length, out_length, scale, kernel, kernel_width, antialiasing
):
    if (scale < 1) and (antialiasing):
        kernel_width = kernel_width / scale

    x = torch.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - kernel_width / 2)
    P = math.ceil(kernel_width) + 2
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(
        0, P - 1, P
    ).view(1, P).expand(out_length, P)

    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = "cubic"

    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing
    )
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing
    )
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = (
            img_aug[0, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        )
        out_1[1, i, :] = (
            img_aug[1, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        )
        out_1[2, i, :] = (
            img_aug[2, idx : idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        )

    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx : idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx : idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx : idx + kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = "cubic"

    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing
    )
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing
    )
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = (
            img_aug[idx : idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        )
        out_1[i, :, 1] = (
            img_aug[idx : idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        )
        out_1[i, :, 2] = (
            img_aug[idx : idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])
        )

    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx : idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx : idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx : idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()


class LRHRDataset(data.Dataset):
    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None
        self.HR_env = None

        if opt["subset_file"] is not None and opt["phase"] == "train":
            with open(opt["subset_file"]) as f:
                self.paths_HR = sorted(
                    [os.path.join(opt["dataroot_HR"], line.rstrip("\n")) for line in f]
                )
            if opt["dataroot_LR"] is not None:
                raise NotImplementedError(
                    "Now subset only supports generating LR on-the-fly."
                )
        else:
            self.HR_env, self.paths_HR = get_image_paths(
                opt["data_type"], opt["dataroot_HR"]
            )
            self.LR_env, self.paths_LR = get_image_paths(
                opt["data_type"], opt["dataroot_LR"]
            )

        assert self.paths_HR, "Error: HR path is empty."
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(
                self.paths_HR
            ), "HR and LR datasets have different number of images - {}, {}.".format(
                len(self.paths_LR), len(self.paths_HR)
            )

        self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt["scale"]
        HR_size = self.opt["HR_size"]

        HR_path = self.paths_HR[index]
        img_HR = read_img(self.HR_env, HR_path)
        if self.opt["phase"] != "train":
            img_HR = modcrop(img_HR, scale)
        if self.opt["color"]:
            img_HR = channel_convert(img_HR.shape[2], self.opt["color"], [img_HR])[0]

        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = read_img(self.LR_env, LR_path)
        else:
            if self.opt["phase"] == "train":
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_HR.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, HR_size)
                W_s = _mod(W_s, random_scale, scale, HR_size)
                img_HR = cv2.resize(
                    np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR
                )
                if img_HR.ndim == 2:
                    img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_HR.shape
            img_LR = imresize_np(img_HR, 1 / scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt["phase"] == "train":
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(
                    np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR
                )
                img_LR = imresize_np(img_HR, 1 / scale, True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR.shape
            LR_size = HR_size // scale

            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h : rnd_h + LR_size, rnd_w : rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[
                rnd_h_HR : rnd_h_HR + HR_size, rnd_w_HR : rnd_w_HR + HR_size, :
            ]

            img_LR, img_HR = augment(
                [img_LR, img_HR], self.opt["use_flip"], self.opt["use_rot"]
            )

        if self.opt["color"]:
            img_LR = channel_convert(C, self.opt["color"], [img_LR])[0]

        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()

        if LR_path is None:
            LR_path = HR_path
        return {"LR": img_LR, "HR": img_HR, "LR_path": LR_path, "HR_path": HR_path}

    def __len__(self):
        return len(self.paths_HR)
