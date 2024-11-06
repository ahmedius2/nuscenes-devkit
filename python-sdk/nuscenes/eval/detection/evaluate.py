# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from typing import List, Dict
from pyquaternion import Quaternion

def get_smooth_egovel(nusc, sample_tkn, target_time_diff_ms=250, global_coords=False):
    sample = nusc.get('sample', sample_tkn)
    sd_tkn = sample['data']['LIDAR_TOP']
    sample_data = nusc.get('sample_data', sd_tkn)
    ts = sample_data['timestamp'] # microseconds

    past_sample_data = sample_data
    past_ts = ts
    while past_sample_data['prev'] != '' and (ts-past_ts) < target_time_diff_ms*1000:
        past_sample_data = nusc.get('sample_data', past_sample_data['prev'])
        past_ts = past_sample_data['timestamp']

    ep = nusc.get('ego_pose', sample_data['ego_pose_token'])
    past_ep = nusc.get('ego_pose', past_sample_data['ego_pose_token'])
    trnsl = np.array(ep['translation'])
    past_trnsl = np.array(past_ep['translation'])
    egovel = (trnsl - past_trnsl) / ((ts - past_ts) * 1e-6)

    if not global_coords:
        rotation = Quaternion(ep['rotation'])
        # Convert the global velocity to ego frame
        egovel = rotation.inverse.rotate(egovel)

    return ep, egovel

class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 det_elapsed_musec: Dict[str,int] = None):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # NOTE the center distances are not important for evaluation but rendering
        # Therefore don't worry to much about it if it is not perfectly right
        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        if det_elapsed_musec is not None:
            print('#############################################')
            print('########## FORECASTED EVALUATION ############')
            print('#############################################')
            # NOTE Do the gt box forecasting here based on detection timestamps
            # det_elapsed_musec_per_sample: given as argument
            # self.gt_boxes is an EvalBoxes object
            sample_tokens = self.gt_boxes.sample_tokens
            for i, sample_tkn in enumerate(sample_tokens):
                tdiff_musec = det_elapsed_musec[sample_tkn]
                drawnow = False #((i % 100) == 0)
                if drawnow:
                    print('Time difference (ms):', tdiff_musec * 1e-3)
                    #_, axes = plt.subplots(1, 3, figsize=(12, 18))
                    ax = self.visualize_sample(sample_tkn) #, axes[0])
                    ep, egovel = get_smooth_egovel(self.nusc, sample_tkn, global_coords=True)
                    ax.arrow(ep['translation'][0], ep['translation'][1], egovel[0], egovel[1],
                                             head_width=0.9, head_length=0.7, fc='red', ec='red')

                self.move_pred_boxes(sample_tkn, tdiff_musec)
                if drawnow:
                    self.visualize_sample(sample_tkn) #, axes[1])
                self.move_gt_boxes(sample_tkn, tdiff_musec)
                if drawnow:
                    self.visualize_sample(sample_tkn) #, axes[2])
                    plt.show()
        # NOTE This seems to be not important for the evaluation as well
        self.sample_tokens = self.gt_boxes.sample_tokens

    def visualize_sample(self, sample_tkn, ax=None):
        nusc = self.nusc
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))
        sample = nusc.get('sample', sample_tkn)
        sensor = 'LIDAR_TOP'
        sample_data = nusc.get('sample_data', sample['data'][sensor])
        pcl_path = os.path.join(self.nusc.dataroot, sample_data['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        pc.filter_xy((-57.6, 57.6), (-57.6, 57.6))

        # point cloud is in lidar coordinate frame, transform it to be in global coordinate frame
        # scatter point cloud on the field in BEV

        ref_cs_record = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        rotation = Quaternion(ref_cs_record['rotation'])
        translation = np.array(ref_cs_record['translation'])
        pc.rotate(rotation.rotation_matrix)
        pc.translate(translation)
        rotation = Quaternion(ego_pose['rotation'])
        translation = np.array(ego_pose['translation'])
        pc.rotate(rotation.rotation_matrix)
        pc.translate(translation)
        ax.scatter(pc.points[0, :], pc.points[1, :], s=0.1, c='blue', alpha=0.3, label='Point Cloud')

        # Draw ground truth boxes in red
        c = ('r', 'r', 'r')
        for gt_box in self.gt_boxes[sample_tkn]:
            Box(gt_box.translation, gt_box.size, Quaternion(gt_box.rotation)).render(ax,
                    colors=c)

        # Draw predicted boxes in green
        c = ('g', 'g', 'g')
        for pred_box in self.pred_boxes[sample_tkn]:
            Box(pred_box.translation, pred_box.size, Quaternion(pred_box.rotation)).render(ax,
                    colors=c)

        # Set labels and show legend
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        #ax.legend(loc='upper right')
        ax.set_aspect('equal', 'box')
        ax.set_title(f"Sample Token: {sample_tkn}")

        return ax


    def move_pred_boxes(self, sample_tkn, tdiff_musec):
        nusc = self.nusc
        sample = nusc.get('sample', sample_tkn)
        sd_tkn = sample['data']['LIDAR_TOP']
        sample_data = nusc.get('sample_data', sd_tkn)
        ts1 = sample_data['timestamp']

        # find the next sample data we can interpolate
        sample_data_next = sample_data
        ts2 = ts1
        while sample_data_next['next'] != '' and ts2 < ts1 + tdiff_musec:
            sample_data_next = nusc.get('sample_data', sample_data_next['next'])
            ts2 = sample_data_next['timestamp']

        tdiff_sec = tdiff_musec * 1e-6
        extrapolate = False
        if ts1 == ts2:
            # coulnt move at all
            sample_data = nusc.get('sample_data', sample_data['prev'])
            ts1 = sample_data['timestamp']
            extrapolate = True
        elif ts2 < ts1 + tdiff_musec:
            # couldnt go further as needed, have to extrapolate
            extrapolate = True

        ep1 = nusc.get('ego_pose', sample_data['ego_pose_token'])
        ep2 = nusc.get('ego_pose', sample_data_next['ego_pose_token'])

        if extrapolate:
            poses_tdiff_sec = ((ts2 - ts1) * 1e-6)
            linear_vel = (np.array(ep2['translation']) - \
                    np.array(ep1['translation'])) / poses_tdiff_sec
            translation_diff = linear_vel * tdiff_sec

            rot1 = Quaternion(ep1['rotation'])
            rot2 = Quaternion(ep2['rotation'])
            angular_velocity = (rot2 * rot1.inverse) ** (1 / poses_tdiff_sec)
            rotation_diff = angular_velocity ** tdiff_sec  #* rot_100ms
        else:
            # we can interpolate between poses
            rratio = tdiff_musec / (ts2 - ts1)
            translation_diff = (np.array(ep2['translation']) - \
                    np.array(ep1['translation'])) * rratio
            rotation_diff = Quaternion.slerp(q0=Quaternion(ep1['rotation']),
                    q1=Quaternion(ep2['rotation']), amount=rratio)

        for box in self.pred_boxes[sample_tkn]:
            box.translation = tuple(np.array(box.translation) + translation_diff)
            #box.rotation = (rotation_diff * Quaternion(box.rotation)).elements

        return

    def move_gt_boxes(self, sample_tkn, tdiff_musec):
        nusc = self.nusc
        ts1 = nusc.get('sample', sample_tkn)['timestamp']
        for gt_box in self.gt_boxes[sample_tkn]:
            sample_anno = nusc.get('sample_annotation', gt_box.sample_anno_token)
            sample_anno_next = sample_anno
            ts2 = ts1
            while sample_anno_next['next'] != '' and ts2 < ts1 + tdiff_musec:
                sample_anno_next = nusc.get('sample_annotation', sample_anno_next['next'])
                ts2 = nusc.get('sample', sample_anno_next['sample_token'])['timestamp']

            tdiff_sec = tdiff_musec * 1e-6
            extrapolate, simple_vel_based_pred = False, False
            if ts1 == ts2:
                # coulnt move at all, just do velocity based prediction
                simple_vel_based_pred = True
            elif ts2 < ts1 + tdiff_musec:
                # couldnt go further as needed, have to extrapolate
                extrapolate = True

            if simple_vel_based_pred:
                if not np.isnan(gt_box.velocity).any():
                    newx = gt_box.translation[0] + gt_box.velocity[0] * tdiff_sec
                    newy = gt_box.translation[1] + gt_box.velocity[1] * tdiff_sec
                    gt_box.translation = (newx, newy, gt_box.translation[2])
            else:
                sa1, sa2 = sample_anno, sample_anno_next
                if extrapolate:
                    poses_tdiff_sec = ((ts2 - ts1) * 1e-6)
                    linear_vel = (np.array(sa2['translation']) - \
                            np.array(sa1['translation'])) / poses_tdiff_sec
                    translation_diff = linear_vel * tdiff_sec

                    rot1 = Quaternion(sa1['rotation'])
                    rot2 = Quaternion(sa2['rotation'])
                    angular_velocity = (rot2 * rot1.inverse) ** (1 / poses_tdiff_sec)
                    rotation_diff = angular_velocity ** tdiff_sec  #* rot_100ms
                else:
                    # we can interpolate between poses
                    rratio = tdiff_musec / (ts2 - ts1)
                    translation_diff = (np.array(sa2['translation']) - \
                            np.array(sa1['translation'])) * rratio
                    rotation_diff = Quaternion.slerp(q0=Quaternion(sa1['rotation']),
                            q1=Quaternion(sa2['rotation']), amount=rratio)

                gt_box.translation = tuple(np.array(gt_box.translation) + translation_diff)
                #gt_box.rotation = (rotation_diff * Quaternion(gt_box.rotation)).elements

        return


    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        do_fine_grained_eval = int(os.getenv('FINE_GRAINED_EVAL', 0))
        sample_npos = {}
        if do_fine_grained_eval > 0:
            print('Doing fine grained eval')
            for gt_box in self.gt_boxes.all:
                if gt_box.detection_name not in sample_npos:
                    sample_npos[gt_box.detection_name] = {}
                inner_d = sample_npos[gt_box.detection_name]
                if gt_box.sample_token not in inner_d:
                    inner_d[gt_box.sample_token] = 1
                else:
                    inner_d[gt_box.sample_token] += 1
            fpath = 'segment_precision_info.json'
            file_exists = os.path.isfile(fpath)
            if file_exists:
                with open(fpath, 'r') as file:
                    segment_precision_info = json.load(file)
            else:
                segment_precision_info = {'fields': ('scene', 'time_segment', 'dist_th', 'class', 
                        'resolution', 'seg_sample_stats'), 'tuples':[]}
        else:
            segment_precision_info = {}

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
 
        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable,
                        dist_th, False, self.nusc, sample_npos, segment_precision_info)
                metric_data_list.set(class_name, dist_th, md)

        if do_fine_grained_eval > 0:
            with open(fpath, 'w') as file:
                json.dump(segment_precision_info, file, indent=2)
        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

        return metrics_summary


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
