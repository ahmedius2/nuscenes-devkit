# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

from typing import Callable, Dict

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData
from nuscenes import NuScenes
import os
import json

ego_vel_cache={}

def get_egopose_and_egovel(nusc, sample, norm=True):
    global ego_vel_cache
    sd_tkn = sample['data']['LIDAR_TOP']
    if (not norm) and sd_tkn in ego_vel_cache:
        return ego_vel_cache[sd_tkn]
    sample_data = nusc.get('sample_data', sd_tkn)
    ep = nusc.get('ego_pose', sample_data['ego_pose_token'])
    # timestamps are in microseconds
    ts = sample_data['timestamp']
    if sample_data['prev'] == '':
        #No prev data, calc speed w.r.t next
        next_sample_data = nusc.get('sample_data', sample_data['next'])
        next_ep = nusc.get('ego_pose', next_sample_data['ego_pose_token'])
        next_ts = next_sample_data['timestamp']
        trnsl = np.array(ep['translation'])
        next_trnsl = np.array(next_ep['translation'])
        egovel = (next_trnsl - trnsl)[:2] / ((next_ts - ts) / 1000000.)
    else:
        prev_sample_data = nusc.get('sample_data', sample_data['prev'])
        prev_ep = nusc.get('ego_pose', prev_sample_data['ego_pose_token'])
        prev_ts = prev_sample_data['timestamp']
        trnsl = np.array(ep['translation'])
        prev_trnsl = np.array(prev_ep['translation'])
        egovel = (trnsl - prev_trnsl)[:2] / ((ts - prev_ts) / 1000000.)

    ego_vel_cache[sd_tkn]=(ep, egovel)
    if norm:
        egovel = np.linalg.norm(egovel)

    return ep, egovel

def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False,
               nusc: NuScenes = None,
               sample_npos : Dict[str,Dict[str,int]] = dict(),
               segment_eval_data = dict()) -> DetectionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    numdets = len(sortind)
    tp = np.zeros(numdets, dtype=bool)  # Accumulator of true positives.
    conf = np.zeros(numdets, dtype=float)  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    trans_err = np.empty(numdets, dtype=float)
    vel_err = np.empty(numdets, dtype=float)
    scale_err = np.empty(numdets, dtype=float)
    orient_err = np.empty(numdets, dtype=float)
    attr_err = np.empty(numdets, dtype=float)
    conf_matched = np.empty(numdets, dtype=float)

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    do_fine_grained_eval = int(os.getenv('FINE_GRAINED_EVAL', 0))
    if len(pred_boxes_list) == 0:
        do_fine_grained_eval = 0

    if do_fine_grained_eval > 0:
        sample_pred_data = {}
        sample_npos_cls = sample_npos[class_name]

        cls_idx = segment_eval_data['class_names'].index(class_name)
        scene_tokens = segment_eval_data['scene_tokens']

    taken = set()  # Initially no gt bounding box is matched.
    num_matched = 0
    for i, ind in enumerate(sortind):
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        if do_fine_grained_eval > 0:
            sample_tkn = pred_box.sample_token
            if pred_box.sample_token not in sample_pred_data:
                sample_pred_data[pred_box.sample_token] = []
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        tp[i] = is_match
        conf[i] = pred_box.detection_score
        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]

            j = num_matched
            trans_err[j] = center_distance(gt_box_match, pred_box)
            vel_err[j] = velocity_l2(gt_box_match, pred_box)
            scale_err[j] = (1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            orient_err[j] = yaw_diff(gt_box_match, pred_box, period=period)
            attr_err[j] = (1 - attr_acc(gt_box_match, pred_box))
            conf_matched[j] = conf[i]
            num_matched += 1

        if do_fine_grained_eval > 0:
            # serialize tp and score only
            sample_pred_data[pred_box.sample_token].append(pred_box.detection_score \
                    if tp[i].item() else -pred_box.detection_score)

    if do_fine_grained_eval > 0:
        data_period_ms = int(os.getenv('DATASET_PERIOD', 100))
        tuples = segment_eval_data['tuples'][class_name][str(dist_th)]
        while sample_pred_data:
            #Get the scene
            sample_tkn = next(iter(sample_pred_data.keys()))
            scene_tkn = nusc.get('sample', sample_tkn)['scene_token']
            scene_idx = scene_tokens[scene_tkn]
            scene = nusc.get('scene', scene_tkn)
            sample_tkn = scene['first_sample_token']
            # For every 1 second, which is 1000ms / 50ms = 20 samples, calc precision
            # Assumes the time between samples is 50 ms
            step_ms = segment_eval_data['segment_time_length_ms']
            msec = 0
            while sample_tkn != "":
                samples_processed = 0
                seg_sample_tokens = []
                seg_num_gt = []
                seg_pred_data = []

                while samples_processed < step_ms//data_period_ms and sample_tkn != "":
                    sample = nusc.get('sample', sample_tkn)
                    #ep, ev = get_egopose_and_egovel(nusc, sample, norm=False)
                    if sample_tkn in sample_pred_data:
                        pred_data = sample_pred_data[sample_tkn]
                        del sample_pred_data[sample_tkn]
                    else:
                        pred_data = list()
                    seg_sample_tokens.append(sample_tkn)
                    seg_num_gt.append(sample_npos_cls.get(sample_tkn, 0))
                    seg_pred_data.append(pred_data)

                    sample = nusc.get('sample', sample_tkn)
                    sample_tkn = sample['next']
                    samples_processed += 1

                tuples.append((
                    scene_idx,
                    msec,
                    seg_sample_tokens,
                    seg_num_gt,
                    seg_pred_data))
                msec += step_ms

    # Check if we have any matches. If not, just return a "no predictions" array.
    if num_matched == 0:
        return DetectionMetricData.no_predictions()

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    fp = np.logical_not(tp)
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    #conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    match_data = {'trans_err': trans_err,
            'vel_err': vel_err,
            'scale_err': scale_err,
            'orient_err': orient_err,
            'attr_err': attr_err,
            'conf': conf_matched}

    for k,v in match_data.items():
        match_data[k] = v[:num_matched]

    for key in match_data.keys():
        # Confidence is used as reference to align with fp and tp. So skip in this step.
        if key != "conf":
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(match_data[key])

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """

    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.
