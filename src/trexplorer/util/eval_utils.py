import numpy as np
import torch
import src.trexplorer.util.misc as utils
from scipy.spatial import cKDTree
import networkx as nx


def calculate_scores(gt_points, pred_points, gt_num_child=None,
                     pred_num_child=None, threshold=1.5):
    gt_tree = cKDTree(gt_points)
    if len(pred_points):
        pred_tree = cKDTree(pred_points)
    else:
        return 0, 0, 0

    dis_gt2pred, dis_gt2pred_idxs = pred_tree.query(gt_points, k=1)
    dis_pred2gt, _ = gt_tree.query(pred_points, k=1)

    min_distances = {}
    for i in range(len(dis_gt2pred_idxs)):
        idx = dis_gt2pred_idxs[i]
        distance = dis_gt2pred[i]
        if idx not in min_distances or distance < min_distances[idx]:
            min_distances[idx] = distance

    filtered_dis_gt2pred_idxs = [idx for idx, _ in min_distances.items()]
    filtered_dis_gt2pred = [min_distances[idx] for idx in filtered_dis_gt2pred_idxs]
    filtered_dis_gt2pred = np.array(filtered_dis_gt2pred)

    # Compute Acc, Recall, F1
    # Bipartite (BP) version does not count duplicate predictions as true positives
    true_positives = [x for x in dis_gt2pred if x < threshold]
    filtered_true_positives = [x for x in filtered_dis_gt2pred if x < threshold]
    recall = len(true_positives) / len(dis_gt2pred)
    recall_bp = len(filtered_true_positives) / len(dis_gt2pred)
    acc = len([x for x in dis_pred2gt if x < threshold]) / len(dis_pred2gt)
    acc_bp = len(filtered_true_positives) / len(dis_pred2gt)
    r_f = 0
    r_f_bp = 0
    if acc * recall:
        r_f = 2 * recall * acc / (acc + recall)
        r_f_bp = 2 * recall_bp * acc_bp / (acc_bp + recall_bp)

    # compute the distance and l1-distance of the num of children for predicted bifurcation points
    if gt_num_child is not None and pred_num_child is not None:
        recalled_points_gt_nchild = [gt_num_child[idx]
                                     for idx, dist in enumerate(dis_gt2pred) if dist < threshold]
        recalled_points_pred_nchild = [pred_num_child[dis_gt2pred_idxs[idx]]
                                       for idx, dist in enumerate(dis_gt2pred) if dist < threshold]
        # L1 distance
        if len(recalled_points_gt_nchild) and len(recalled_points_pred_nchild):
            arr1 = np.array(recalled_points_gt_nchild)
            arr2 = np.array(recalled_points_pred_nchild)
            nchild_dist = np.sum(arr1 - arr2) / len(recalled_points_gt_nchild)
            nchild_dist_l1 = np.sum(np.abs(arr1 - arr2)) / len(recalled_points_gt_nchild)
        else:
            nchild_dist, nchild_dist_l1 = np.nan, np.nan
    else:
        return acc, acc_bp, recall, recall_bp, r_f, r_f_bp

    return acc, acc_bp, recall, recall_bp, r_f, r_f_bp, nchild_dist_l1, nchild_dist


def get_score_nx(preds, targets, elapsed_time, dist=False):
    data_dict = {'preds_nchild_list': [],
                 'targets_nchild_list': [],
                 'preds_bifur_list': [],
                 'targets_bifur_list': [],
                 'preds_list': [],
                 'targets_list': [],}

    for i, (pred, target) in enumerate(zip(preds, targets)):
        for item in data_dict:
            data_dict[item].append([])

        data_dict['preds_list'][i] = list(nx.get_node_attributes(pred, 'position').values())
        out_degrees = dict(pred.out_degree())
        bifur_nodes = [node for node in pred.nodes if out_degrees[node] > 1]
        data_dict['preds_bifur_list'][i] = [pred.nodes[node]['position'] for node in bifur_nodes]
        data_dict['preds_nchild_list'][i] = [out_degrees[node] for node in bifur_nodes]

        data_dict['targets_list'][i] = list(nx.get_node_attributes(target, 'position').values())
        out_degrees = dict(target.out_degree())
        bifur_nodes = [node for node in target.nodes if out_degrees[node] > 1]
        data_dict['targets_bifur_list'][i] = [target.nodes[node]['position'] for node in bifur_nodes]
        data_dict['targets_nchild_list'][i] = [out_degrees[node] for node in bifur_nodes]

    scores_dict = {'total_points': [],
                   'total_bifur_points': [],
                   'num_bifur_points': [],
                   'num_points': [],
                   'bifur_scores': [],
                   'scores': [], }

    for (pred_points, target_points) in zip(data_dict['preds_list'], data_dict['targets_list']):
        scores_dict['scores'].append(list(calculate_scores(target_points, pred_points)))
        scores_dict['num_points'].append(len(pred_points))
        scores_dict['total_points'].append(len(target_points))

    for (pred_points, target_points, pred_nchild, target_nchild) in (
            zip(data_dict['preds_bifur_list'], data_dict['targets_bifur_list'], data_dict['preds_nchild_list'], data_dict['targets_nchild_list'])):
        if len(pred_points) and len(target_points):
            scores_dict['bifur_scores'].append(list(calculate_scores(target_points, pred_points, target_nchild, pred_nchild, threshold=3.5)))
        else:
            scores_dict['bifur_scores'].append([0, 0, 0, 0, 0, 0, np.nan, np.nan])
        scores_dict['num_bifur_points'].append(len(pred_points))
        scores_dict['total_bifur_points'].append(len(target_points))

    stats_t = {'num_points': torch.FloatTensor(scores_dict['num_points']).to('cuda')}
    stats_t['scores'] = torch.FloatTensor(scores_dict['scores']).to('cuda')
    stats_t['total_points'] = torch.FloatTensor(scores_dict['total_points']).to('cuda')
    stats_t['num_bifur_points'] = torch.FloatTensor(scores_dict['num_bifur_points']).to('cuda')
    stats_t['bifur_scores'] = torch.FloatTensor(scores_dict['bifur_scores']).to('cuda')
    stats_t['total_bifur_points'] = torch.FloatTensor(scores_dict['total_bifur_points']).to('cuda')
    stats_t['elapsed_time'] = torch.FloatTensor(elapsed_time).to('cuda')

    if dist:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        stats_reduced = utils.gather_stats(stats_t)
    else:
        stats_reduced = stats_t

    stats_reduced['avg_num_points'] = torch.mean(stats_reduced['num_points'], dim=0)
    stats_reduced['avg_scores'] = torch.mean(stats_reduced['scores'], dim=0)
    stats_reduced['avg_total_points'] = torch.mean(stats_reduced['total_points'], dim=0)
    stats_reduced['avg_num_bifur_points'] = torch.mean(stats_reduced['num_bifur_points'], dim=0)
    stats_reduced['avg_bifur_scores'] = torch.nanmean(stats_reduced['bifur_scores'], dim=0)
    stats_reduced['avg_total_bifur_points'] = torch.mean(stats_reduced['total_bifur_points'], dim=0)
    stats_reduced['avg_elapsed_time'] = torch.mean(stats_reduced['elapsed_time'], dim=0)

    return stats_reduced


def get_empty_stats_dict():
    stats_t = {'avg_num_points': [], 'avg_scores': [], 'avg_total_points': [],
               'avg_num_bifur_points': [], 'avg_bifur_scores': [], 'avg_total_bifur_points': []}

    return stats_t


def get_stats_message(stats):
    message = "All points\n"
    message += (f"Acc: {stats['avg_scores'][0]:.3f} \t "
                f"| \t Acc-BP: {stats['avg_scores'][1]:.3f} \t "
                f"| \t Recall: {stats['avg_scores'][2]:.3f} \t "
                f"| \t Recall-BP: {stats['avg_scores'][3]:.3f} \t "
                f"| \t F1: {stats['avg_scores'][4]:.3f} \t "
                f"| \t F1-BP: {stats['avg_scores'][5]:.3f} \n ")
    message += (f"| \t Avg Points: {int(stats['avg_num_points'].item())} \t "
                f"| \t Tot Points: {int(stats['avg_total_points'].item())} \t "
                f"| \t Avg Time: {stats['avg_elapsed_time'].item():.3f}\n")

    message += "Bifur points\n"
    message += (f"Acc: {stats['avg_bifur_scores'][0]:.3f} \t "
                f"| \t Acc-BP: {stats['avg_bifur_scores'][1]:.3f} \t "
                f"| \t Recall: {stats['avg_bifur_scores'][2]:.3f} \t "
                f"| \t Recall-BP: {stats['avg_bifur_scores'][3]:.3f} \t "
                f"| \t F1: {stats['avg_bifur_scores'][4]:.3f} \t "
                f"| \t F1-BP: {stats['avg_bifur_scores'][5]:.3f} \n "
                f"| \t NC Error: {stats['avg_bifur_scores'][6]:.3f} \t "
                f"| \t Avg Points: {int(stats['avg_num_bifur_points'].item())} \t "
                f"| \t Tot Points: {int(stats['avg_total_bifur_points'].item())} \t "
                f"| \t Avg Time: {stats['avg_elapsed_time'].item():.3f}\n")

    return message


def get_avg_stats_message(stats):
    message = "All points\n"
    message += (f"Acc: {stats['avg_scores'][0]:.3f} \t "
                f"| \t Acc-BP: {stats['avg_scores'][1]:.3f} \t "
                f"| \t Recall: {stats['avg_scores'][2]:.3f} \t "
                f"| \t Recall-BP: {stats['avg_scores'][3]:.3f} \t "
                f"| \t F1: {stats['avg_scores'][4]:.3f} \t "
                f"| \t F1-BP: {stats['avg_scores'][5]:.3f} \n ")
    message += (f"| \t Avg Points: {int(stats['avg_num_points'].item())} \t "
                f"| \t Tot Points: {int(stats['avg_total_points'].item())} \t "
                f"| \t Avg Time: {stats['avg_elapsed_time'].item():.3f}\n")

    message += "Bifur points\n"
    message += (f"Acc: {stats['avg_bifur_scores'][0]:.3f} \t "
                f"| \t Acc-BP: {stats['avg_bifur_scores'][1]:.3f} \t "
                f"| \t Recall: {stats['avg_bifur_scores'][2]:.3f} \t "
                f"| \t Recall-BP: {stats['avg_bifur_scores'][3]:.3f} \t "
                f"| \t F1: {stats['avg_bifur_scores'][4]:.3f} \t "
                f"| \t F1-BP: {stats['avg_bifur_scores'][5]:.3f} \n "
                f"| \t NC Error: {stats['avg_bifur_scores'][6]:.3f} \t "
                f"| \t Avg Points: {int(stats['avg_num_bifur_points'].item())} \t "
                f"| \t Tot Points: {int(stats['avg_total_bifur_points'].item())} \t "
                f"| \t Avg Time: {stats['avg_elapsed_time'].item():.3f}\n")

    return message