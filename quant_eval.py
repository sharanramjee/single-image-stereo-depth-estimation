import os
import numpy as np
from PIL import Image


def replace_zeros(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def compute_errors(gt_path, pred_path):
    gt = np.clip(np.asarray(Image.open(gt_path), dtype=float) / 255, 0, 1)
    pred = np.clip(np.asarray(Image.open(pred_path), dtype=float) / 255, 0, 1)
    gt = replace_zeros(gt)
    pred = replace_zeros(pred)
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_avg_errors(gt_dir, pred_dir):
    gts = [gt_dir+f for f in os.listdir(gt_dir) if not f.startswith('.')]
    preds = [pred_dir+f for f in os.listdir(pred_dir) if not f.startswith('.')]
    abs_rels = list()
    sq_rels = list()
    rmses = list()
    rmse_logs = list()
    count = 1
    for gt, pred in zip(gts, preds):
        abs_rel, sq_rel, rmse, rmse_log, _, _, _ = compute_errors(gt, pred)
        abs_rels.append(abs_rel)
        sq_rels.append(sq_rel)
        rmses.append(rmse)
        rmse_logs.append(rmse_log)
        print('Image', count, 'processed')
        count += 1
    avg_abs_rel = np.mean(abs_rels)
    print('ARD:', avg_abs_rel)
    avg_sq_rel = np.mean(sq_rels)
    print('SRD:', avg_sq_rel)
    avg_rmse = np.mean(rmses)
    print('RMSE:', avg_rmse)
    avg_rmse_log = np.mean(rmse_logs)
    print('log RMSE:', avg_rmse_log)


if __name__ == '__main__':
    gt_dir_path = 'data/test/depth/'
    pred_dir_path = 'output/DenseDepth_original/'
    # pred_dir_path = 'output/stereo_depth_estimator/'
    compute_avg_errors(gt_dir_path, pred_dir_path)
