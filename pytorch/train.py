import os.path
import argparse
import torch

from pytorch.data import LRHRDataset
from pytorch.utils import *
from pytorch.model import SRGANModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to option JSON file."
    )
    opt = parse(parser.parse_args().opt, is_train=True)
    opt = dict_to_nonedict(opt)

    if opt["path"]["resume_state"]:
        resume_state = torch.load(opt["path"]["resume_state"])
    else:
        resume_state = None
        mkdir_and_rename(opt["path"]["experiments_root"])
        mkdirs(
            (
                path
                for key, path in opt["path"].items()
                if not key == "experiments_root"
                and "pretrain_model" not in key
                and "resume" not in key
            )
        )

    setup_logger(None, opt["path"]["log"], "train", level=logging.INFO, screen=True)
    setup_logger("val", opt["path"]["log"], "val", level=logging.INFO)
    logger = logging.getLogger("base")

    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        check_resume(opt)

    logger.info(dict2str(opt))

    seed = opt["train"]["manual_seed"]
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(seed))
    set_random_seed(seed)

    torch.backends.cudnn.benckmark = True

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = LRHRDataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            logger.info(
                "Number of train images: {:,d}, iters: {:,d}".format(
                    len(train_set), train_size
                )
            )
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info(
                "Total epochs needed: {:d} for iters {:,d}".format(
                    total_epochs, total_iters
                )
            )
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == "val":
            val_set = LRHRDataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info(
                "Number of val images in [{:s}]: {:d}".format(
                    dataset_opt["name"], len(val_set)
                )
            )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None

    model = SRGANModel(opt)

    if resume_state:
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)
    else:
        current_step = 0
        start_epoch = 0

    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )
    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            model.update_learning_rate()

            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                logger.info(message)

            if current_step % opt["train"]["val_freq"] == 0:
                avg_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(
                        os.path.basename(val_data["LR_path"][0])
                    )[0]
                    img_dir = os.path.join(opt["path"]["val_images"], img_name)
                    mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = tensor2img(visuals["SR"])
                    gt_img = tensor2img(visuals["HR"])

                    save_img_path = os.path.join(
                        img_dir, "{:s}_{:d}.png".format(img_name, current_step)
                    )
                    save_img(sr_img, save_img_path)

                    crop_size = opt["scale"]
                    gt_img = gt_img / 255.0
                    sr_img = sr_img / 255.0
                    cropped_sr_img = sr_img[
                        crop_size:-crop_size, crop_size:-crop_size, :
                    ]
                    cropped_gt_img = gt_img[
                        crop_size:-crop_size, crop_size:-crop_size, :
                    ]
                    avg_psnr += calculate_psnr(
                        cropped_sr_img * 255, cropped_gt_img * 255
                    )

                avg_psnr = avg_psnr / idx

                logger.info("# Validation # PSNR: {:.4e}".format(avg_psnr))
                logger_val = logging.getLogger("val")
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}".format(
                        epoch, current_step, avg_psnr
                    )
                )

            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                logger.info("Saving models and training states.")
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info("Saving the final model.")
    model.save("latest")
    logger.info("End of training.")


if __name__ == "__main__":
    main()
