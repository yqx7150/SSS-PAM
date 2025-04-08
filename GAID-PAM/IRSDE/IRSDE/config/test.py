from data.util import bgr2ycbcr
from data import create_dataloader, create_dataset
import utils as util
import argparse
import logging
import math
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils
import os.path as osp

import numpy as np
import torch
from IPython import embed
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")

# options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True,
                    help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)


results_root = util.mkdir_and_rename(
    opt["path"]["results_root"]
)  # rename experiment folder if exists

opt["path"]["results_root"] = results_root
opt["path"]["log"] = osp.join(results_root, 'log')

# mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"],
                 eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
lpips_fn = lpips.LPIPS(net='alex').to(device)

scale = opt['degradation']['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    # todo
    test_results["psnr_LQ"] = []
    test_results["ssim_LQ"] = []

    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    test_results["lpips"] = []
    test_results["lpips_LQ"] = []
    test_times = []

    for i, test_data in enumerate(test_loader):
        single_img_psnr = []
        single_img_ssim = []
        single_img_psnr_y = []
        single_img_ssim_y = []

        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # input dataset_LQ
        LQ, GT = test_data["LQ"], test_data["GT"]
        noisy_state = sde.noise_state(LQ)

        model.feed_data(noisy_state, LQ, GT)
        tic = time.time()
        model.test(sde, save_states=True,
                   results_root=opt["path"]["results_root"])
        toc = time.time()
        test_times.append(toc - tic)
        print(toc-tic, '---------------')

        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        print(SR_img.shape)
        output = util.tensor2img(SR_img.squeeze())  # uint8

        LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
        GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8

        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(
                dataset_dir, "jieguo", img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(
                dataset_dir, "jieguo", img_name + ".png")
        if not os.path.exists(os.path.join(dataset_dir, "jieguo")):
            os.makedirs(os.path.join(dataset_dir, "jieguo"), exist_ok=True)
        util.save_img(output, save_img_path)

        LQ_img_path = os.path.join(dataset_dir, "LQ/", img_name + "_LQ.png")
        if not os.path.exists(os.path.join(dataset_dir, "LQ")):
            os.makedirs(os.path.join(dataset_dir, "LQ"), exist_ok=True)
        print("lQ_path: ", LQ_img_path)
        GT_img_path = os.path.join(dataset_dir, "HQ/", img_name + "_HQ.png")
        if not os.path.exists(os.path.join(dataset_dir, "HQ")):
            os.makedirs(os.path.join(dataset_dir, "HQ"), exist_ok=True)
        print("GT_path: ", GT_img_path)
        util.save_img(LQ_, LQ_img_path)
        util.save_img(GT_, GT_img_path)

        if need_GT:
            gt_img = GT_ / 255.0
            sr_img = output / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else scale
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]
                cropped_gt_img = gt_img[
                    crop_border:-crop_border, crop_border:-crop_border
                ]

            psnr = util.calculate_psnr(
                cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(
                cropped_sr_img * 255, cropped_gt_img * 255)

            lp_score = lpips_fn(
                GT.to(device) * 2 - 1, SR_img.to(device) * 2 - 1).squeeze().item()
            lp_score_LQ = lpips_fn(
                GT.to(device) * 2 - 1, LQ.to(device) * 2 - 1).squeeze().item()

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lp_score)
            test_results["lpips_LQ"].append(lp_score_LQ)

            if len(gt_img.shape) == 3:
                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    if crop_border == 0:
                        cropped_sr_img_y = sr_img_y
                        cropped_gt_img_y = gt_img_y
                    else:
                        cropped_sr_img_y = sr_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                        cropped_gt_img_y = gt_img_y[
                            crop_border:-crop_border, crop_border:-crop_border
                        ]
                    psnr_y = util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    ssim_y = util.calculate_ssim(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )

                    test_results["psnr_y"].append(psnr_y)
                    test_results["ssim_y"].append(ssim_y)

                    logger.info(
                        "img{:3d}:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                            i, img_name, psnr, ssim, lp_score, psnr_y, ssim_y
                        )
                    )
            else:
                # result-GT
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}.".format(
                        img_name, psnr, ssim
                    )
                )
                # LQ-GT
                lq_psnr = util.calculate_psnr(LQ_, GT_)
                lq_ssim = util.calculate_ssim(LQ_, GT_)
                logger.info(
                    "LQ-GT - PSNR: {:.6f} dB; SSIM: {:.6f};".format(
                        lq_psnr, lq_ssim
                    )
                )
                test_results['psnr_LQ'].append(lq_psnr)
                test_results['ssim_LQ'].append(lq_ssim)

        else:
            logger.info(img_name)

    ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
    ave_lpips_LQ = sum(test_results["lpips_LQ"]) / \
        len(test_results["lpips_LQ"])
    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
    ave_psnr_LQ = sum(test_results['psnr_LQ']) / len(test_results['psnr_LQ'])
    ave_ssim_LQ = sum(test_results['ssim_LQ']) / len(test_results['ssim_LQ'])

    std_psnr = np.std(test_results["psnr"], ddof=1)
    std_ssim = np.std(test_results["ssim"], ddof=1)
    std_psnr_LQ = np.std(test_results['psnr_LQ'], ddof=1)
    std_ssim_LQ = np.std(test_results['ssim_LQ'], ddof=1)

    logger.info(
        "【results-GT】----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr, ave_ssim
        )
    )

    logger.info(
        "【LQ-GT】----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, ave_psnr_LQ, ave_ssim_LQ
        )
    )

    logger.info(
        "【results-GT】----std PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, std_psnr, std_ssim
        )
    )
    logger.info(
        "【LQ-GT】----std PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n".format(
            test_set_name, std_psnr_LQ, std_ssim_LQ
        )
    )

    logger.info(
        "【results-GT】----average LPIPS\t: {:.6f}\n".format(ave_lpips)
    )
    logger.info(
        "【LQ-GT】----average LPIPS\t: {:.6f}\n".format(ave_lpips_LQ)
    )

    print(f"average test time: {np.mean(test_times):.4f}")
    
    import cal_psnr_for_iter
    cal_psnr_for_iter.cal_psnr_for_iter_func(opt["path"]["results_root"])

