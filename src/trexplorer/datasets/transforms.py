import pickle
import numpy as np
import torch
import re
from bigtree import find_name, levelordergroup_iter, Node
from monai.utils import convert_to_tensor, pytorch_after
from monai.transforms import (
    Transform,
    MapTransform,
    Randomizable,
    Crop,
    Pad,
    LoadImage,
    ScaleIntensity)


class LoadAnnotPickle(Transform):
    def __init__(self):
        pass

    def __call__(self, input):
        with open(input, 'rb') as handle:
            data = pickle.load(handle)
        data['index'] = [int(s) for s in re.findall(r'\d+', input)][-1]
        return data


class LoadAnnotPickled(MapTransform):
    def __init__(self, keys,
                 allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.transform = LoadAnnotPickle()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])
        return d


class CropAndPad(Transform):
    def __init__(self, sub_vol_size):
        self.crop = Crop()
        self.pad = Pad()
        self.sub_vol_size = sub_vol_size

    @staticmethod
    def compute_slices(roi_center, roi_size, img_size):
        """
        Compute the crop slices based on specified `center & size` or `start & end` or `slices`.
        """
        assert roi_center is not None and roi_size is not None and img_size is not None
        img_size_t = convert_to_tensor(data=img_size, dtype=torch.int16, wrap_sequence=True, device="cuda")
        roi_center_t = convert_to_tensor(data=roi_center, dtype=torch.int16, wrap_sequence=True, device="cuda")
        roi_size_t = convert_to_tensor(data=roi_size, dtype=torch.int16, wrap_sequence=True, device="cuda")
        _zeros = torch.zeros_like(roi_center_t)
        half = (
            torch.divide(roi_size_t, 2, rounding_mode="floor")
            if pytorch_after(1, 8)
            else torch.floor_divide(roi_size_t, 2)
        )
        roi_start_t = roi_center_t - half
        roi_end_t = roi_start_t + roi_size_t
        roi_start_clipped_t = torch.maximum(roi_start_t, _zeros)
        roi_end_clipped_t = torch.minimum(roi_end_t, img_size_t)
        roi_start_pad_t = torch.abs(torch.minimum(roi_start_t, _zeros))
        roi_end_pad_t = torch.maximum(roi_end_t - img_size_t, _zeros)

        slices = [slice(int(s), int(e)) for s, e in zip(roi_start_clipped_t.tolist(), roi_end_clipped_t.tolist())]
        padding = [(0, 0)] + [(int(s), int(e)) for s, e in zip(roi_start_pad_t.tolist(), roi_end_pad_t.tolist())]

        return slices, padding

    def extract_image_crop(self, image_data, node_position, image_min):
        crop_slices, crop_padding = self.compute_slices(node_position,
                                                        self.sub_vol_size,
                                                        image_data.shape[-3:])
        cropped_image = self.crop(image_data, crop_slices)
        cropped_image = self.pad(cropped_image, crop_padding, value=image_min)

        return cropped_image

    def __call__(self, data, position, image_min):
        # extract image sub volume around root point
        data = self.extract_image_crop(data, position, image_min)

        return data


class CropAndPadd(MapTransform):
    def __init__(self, keys, sub_vol_size):
        super().__init__(keys)
        self.transform = CropAndPad(sub_vol_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key == 'image':
                d[key] = self.transform(d[key], d['label']['root_position'], -1.0)
            elif key == 'mask':
                d[key] = self.transform(d[key], d['label']['root_position'], 0.0)
            else:
                raise NotImplementedError

        return d


class ExtRandomSubTree(Randomizable, Transform):
    def __init__(self, root_prob, bifur_prob, end_prob, seq_len, num_prev_pos):
        self.root_prob = root_prob
        self.bifur_prob = bifur_prob
        self.end_prob = end_prob
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos

    def randomize(self, data):
        """
        Pick a random point using curated sampling
        """
        prob = self.R.random()

        if prob < self.bifur_prob:
            cumulative_index = -1
            index_mapping = {}
            for tree_id, lst in enumerate(data['bifur_ids']):
                for index_within_list, value in enumerate(lst):
                    cumulative_index += 1
                    index_mapping[cumulative_index] = [tree_id, index_within_list]
            point_type = 'bifurcation'
            idx = self.R.randint(low=0, high=cumulative_index, size=1)[0]
            tree_id, point_id_index = index_mapping[idx]
            point_id = data['bifur_ids'][tree_id][point_id_index]
            dist = self.R.randint(low=0, high=self.seq_len, size=1)[0]
            selected_node = self.select_nearby_point(data['branches'][tree_id], point_id, dist)

        elif prob < (self.bifur_prob + self.end_prob):
            cumulative_index = -1
            index_mapping = {}
            for tree_id, lst in enumerate(data['endpts_ids']):
                for index_within_list, value in enumerate(lst):
                    cumulative_index += 1
                    index_mapping[cumulative_index] = [tree_id, index_within_list]
            point_type = 'end'
            idx = self.R.randint(low=0, high=cumulative_index, size=1)[0]
            tree_id, point_id_index = index_mapping[idx]
            point_id = data['endpts_ids'][tree_id][point_id_index]
            dist = self.R.randint(low=0, high=self.seq_len, size=1)[0]
            selected_node = self.select_nearby_point(data['branches'][tree_id], point_id, dist)

        elif prob < (self.bifur_prob + self.end_prob + self.root_prob):
            tree_id = self.R.randint(low=0, high=len(data['branches']), size=1)[0]
            full_tree = data['branches'][tree_id]
            point_type = 'root'
            point_id = full_tree.node_name
            dist = self.R.randint(low=0, high=self.seq_len, size=1)[0]
            selected_node = full_tree
            for i, level in enumerate(levelordergroup_iter(full_tree)):
                if i == dist:
                    point_index = self.R.randint(low=0, high=len(level), size=1)[0]
                    selected_node = level[point_index]
                    break

        else:
            tree_id = self.R.randint(low=0, high=len(data['branches']), size=1)[0]
            full_tree = data['branches'][tree_id]
            dist = 0
            point_type = 'random'
            point_id_index = self.R.randint(low=0, high=len(data['all_ids'][tree_id]), size=1)[0]
            point_id = data['all_ids'][tree_id][point_id_index]
            selected_node = find_name(full_tree, point_id)

        assert selected_node is not None, "Selected node not found!"
        return tree_id, selected_node, point_id, point_type, dist

    @staticmethod
    def select_nearby_point(full_tree, point_id, dist):
        """
        Pick a point "dist" up from the point_id.
        """
        selected_node = find_name(full_tree, point_id)
        for i in range(dist):
            parent = selected_node.parent
            if parent is not None:
                selected_node = parent
            else:
                continue

        return selected_node

    @staticmethod
    def get_past_traj_start(selected_node, num_prev_pos):
        past_traj_head_node = selected_node
        for past_nodes_num in range(num_prev_pos):
            if past_traj_head_node.parent is not None:
                past_traj_head_node = past_traj_head_node.parent
            else:
                break

        return past_traj_head_node, past_nodes_num

    @staticmethod
    def create_subtree_with_past_tr(selected_node, past_traj_num, seq_len):
        root_node = Node(selected_node.name,
                         position=selected_node.position,
                         radius=selected_node.radius,
                         label=selected_node.label)

        for i, level in enumerate(levelordergroup_iter(selected_node)):
            if i == 0:
                continue
            elif i <= seq_len:
                for node in level:
                    parent_node = find_name(root_node, node.parent.name)
                    Node(node.name,
                         position=node.position,
                         radius=node.radius,
                         parent=parent_node,
                         label=node.label)
            else:
                break

        if selected_node.parent is not None:
            current_node = selected_node.parent
            past_tr_end_node = Node(current_node.name,
                                    position=current_node.position,
                                    radius=current_node.radius,
                                    label=current_node.label)

            past_tr_head_node = past_tr_end_node
            for i in range(1, past_traj_num):
                current_node = current_node.parent
                past_tr_head_node.parent = Node(current_node.name,
                                                position=current_node.position,
                                                radius=current_node.radius,
                                                label=current_node.label)
                past_tr_head_node = past_tr_head_node.parent
            root_node.parent = past_tr_end_node
        else:
            past_tr_head_node = root_node

        return root_node, past_tr_head_node

    def __call__(self, data):
        tree_id, selected_node, point_id, point_type, dist = self.randomize(data)
        _, past_traj_num = self.get_past_traj_start(selected_node, self.num_prev_pos)
        selected_node, past_traj_head_node = self.create_subtree_with_past_tr(selected_node,
                                                                              past_traj_num, self.seq_len)

        data = {'index': data['index'],
                'tree_id': tree_id,
                'point_id': point_id,
                'point_type': point_type,
                'dist': dist,
                'selected_node': selected_node,
                'past_traj_head_node': past_traj_head_node
                }

        return data


class ExtRandomSubTreed(Randomizable, MapTransform):
    def __init__(self, keys, root_prob, bifur_prob, end_prob, seq_len, num_prev_pos):
        super(Randomizable, self).__init__(keys)
        self.transform = ExtRandomSubTree(root_prob, bifur_prob, end_prob, seq_len, num_prev_pos)

    def set_random_state(self, seed=None, state=None):
        self.transform.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class ConvertTreeToTargets(Transform):
    def __init__(self, seq_len, num_prev_pos, sub_vol_size, class_dict):
        self.seq_len = seq_len
        self.num_prev_pos = num_prev_pos
        self.sub_vol_size = sub_vol_size
        self.class_dict = class_dict

    def extract_sub_tree(self, selected_node):
        root_position = np.asarray(selected_node.position).astype(int)
        grouped_node_seq = self.init_sub_tree(root_position)
        grouped_node_seq, level = self.extract_ordered_sub_tree(grouped_node_seq,
                                                                selected_node, root_position)
        grouped_node_seq = self.pad_sub_tree(grouped_node_seq, level)

        return grouped_node_seq

    @staticmethod
    def init_sub_tree(root_position):
        grouped_node_seq = {'root_position': root_position,
                            'node_ids': [], 'node_parents': [], 'parent_idxs': [], 'rel_positions': [],
                            'radii': [], 'labels': []}

        return grouped_node_seq

    @staticmethod
    def find_element_index(element, my_list):
        try:
            index = my_list.index(element)
            return index
        except ValueError:
            return -1

    def extract_ordered_sub_tree(self, grouped_node_seq, selected_node, root_position):
        prev_level_branches = []
        ordered_branches = []
        for level, curr_level_nodes in enumerate(levelordergroup_iter(selected_node)):
            if level < self.seq_len:
                new_branches, curr_level_branches = self.find_branches(curr_level_nodes, prev_level_branches)
                if len(new_branches):
                    ordered_branches += new_branches

                grouped_node_seq = self.init_prop_lists(grouped_node_seq)
                for index_br, branch in enumerate(ordered_branches):
                    curr_level_br_index = self.find_element_index(branch, curr_level_branches)
                    if curr_level_br_index == -1:
                        grouped_node_seq = self.pad_finished_branch(grouped_node_seq, branch, index_br, level)
                    else:
                        grouped_node_seq = self.append_node_info(grouped_node_seq, curr_level_nodes[curr_level_br_index],
                                                                 root_position, level)
                prev_level_branches = [name.split("-")[0] for name in grouped_node_seq['node_ids'][level]]
            else:
                level -= 1
                break

        return grouped_node_seq, level

    @staticmethod
    def find_branches(curr_level_nodes, prev_level_branches):
        curr_level_branches = []
        for node in curr_level_nodes:
            curr_level_branches.append(node.node_name.split("-")[0])
        new_branches = sorted(list(set(curr_level_branches) - set(prev_level_branches)))

        return new_branches, curr_level_branches

    @staticmethod
    def init_prop_lists(grouped_node_seq):
        grouped_node_seq['node_ids'].append([])
        grouped_node_seq['node_parents'].append([])
        grouped_node_seq['parent_idxs'].append([])
        grouped_node_seq['rel_positions'].append([])
        grouped_node_seq['radii'].append([])
        grouped_node_seq['labels'].append([])

        return grouped_node_seq

    def append_node_info(self, grouped_node_seq, node, root_position, level):
        rel_position = np.array(node.position - root_position, dtype=float)
        node_radius = node.radius
        num_children = len(node.children)
        if num_children < 2:
            label_id = num_children
        else:
            label_id = self.class_dict['bifurcation']
        rel_position /= (self.sub_vol_size // 2)
        node_radius /= (self.sub_vol_size // 2)

        if level:
            parent_id = node.parent.node_name if node.parent is not None else -1
            parent_idxs = self.find_element_index(parent_id, grouped_node_seq['node_ids'][level - 1])
        else:
            parent_id = -1
            parent_idxs = -1

        grouped_node_seq['node_ids'][level].append(node.node_name)
        grouped_node_seq['node_parents'][level].append(parent_id)
        grouped_node_seq['parent_idxs'][level].append(parent_idxs)
        grouped_node_seq['rel_positions'][level].append(rel_position.tolist())
        grouped_node_seq['radii'][level].append(node_radius)
        grouped_node_seq['labels'][level].append(label_id)

        return grouped_node_seq

    def pad_finished_branch(self, grouped_node_seq, branch_nm, index_br, level):
        parent_nm = grouped_node_seq['node_ids'][level - 1][index_br]
        point_nm = parent_nm.split("-")[1]
        new_point_nm = str(int(point_nm) + 1)
        node_id = branch_nm + "-" + new_point_nm

        node_position = grouped_node_seq['rel_positions'][level - 1][index_br]
        node_label = self.class_dict['background']
        grouped_node_seq['node_ids'][level].append(node_id)
        grouped_node_seq['node_parents'][level].append(parent_nm)
        grouped_node_seq['parent_idxs'][level].append(index_br)
        grouped_node_seq['rel_positions'][level].append(node_position)
        grouped_node_seq['radii'][level].append(0.0)
        grouped_node_seq['labels'][level].append(node_label)

        return grouped_node_seq

    def pad_sub_tree(self, grouped_node_seq, level):
        if level < self.seq_len - 1:
            prev_level_names = grouped_node_seq['node_ids'][-1]
            for j in range(level + 1, self.seq_len):
                level_names = []
                for name in prev_level_names:
                    br, nm = name.split("-")
                    new_nm = str(int(nm) + 1)
                    level_names.append(br + "-" + new_nm)
                level_positions = grouped_node_seq['rel_positions'][-1]
                node_label = self.class_dict['background']
                labels = (np.zeros(np.array(grouped_node_seq['labels'][-1]).shape, dtype=int) + node_label).tolist()
                level_radii = np.zeros(np.array(grouped_node_seq['radii'][-1]).shape).tolist()

                grouped_node_seq['node_ids'].append(level_names)
                grouped_node_seq['node_parents'].append(prev_level_names)
                grouped_node_seq['parent_idxs'].append(grouped_node_seq['parent_idxs'][-1])
                grouped_node_seq['rel_positions'].append(level_positions)
                grouped_node_seq['radii'].append(level_radii)
                grouped_node_seq['labels'].append(labels)
                prev_level_names = level_names

        return grouped_node_seq

    def generate_past_trajectory_pp(self, root_node, root_position):
        past_traj_label = []
        parent = root_node

        for i in range(self.num_prev_pos):
            node_info_list = []
            node_position = np.asarray(parent.position)
            rel_position = (node_position - root_position).astype(float)
            rel_position /= (self.sub_vol_size // 2)
            node_info_list.append(rel_position)
            node_radius = np.asarray(parent.radius).reshape(1)
            node_radius /= (self.sub_vol_size // 2)
            node_info_list.append(node_radius)
            past_traj_label.append(np.concatenate(node_info_list))
            parent = parent.parent
            if parent is None:
                break

        past_traj_label_np = np.asarray(past_traj_label)

        num_missing_points = self.num_prev_pos - past_traj_label_np.shape[0]
        if num_missing_points:
            last_point_info = past_traj_label_np[-1:]
            padding_info = np.tile(last_point_info, (num_missing_points, 1))
            past_traj_label_np = np.vstack((past_traj_label_np, padding_info))

        past_traj_label_t = torch.FloatTensor(past_traj_label_np).flatten()

        return past_traj_label_t

    def __call__(self, data):
        data.update(self.extract_sub_tree(data['selected_node']))
        data['past_tr'] = self.generate_past_trajectory_pp(data['selected_node'], data['root_position'])
        data['index'] = data['index']
        data.pop('selected_node')
        data.pop('past_traj_head_node')

        return data


class ConvertTreeToTargetsd(MapTransform):
    def __init__(self, keys, seq_len, num_prev_pos, sub_vol_size, class_dict):
        super().__init__(keys)
        self.transform = ConvertTreeToTargets(seq_len, num_prev_pos, sub_vol_size, class_dict)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])

        return d


class LoadImageCropsAndTrees(Transform):
    def __init__(self, seq_len, num_prev_pos, sub_vol_size, class_dict, images, annots, masks):
        self.images = images
        self.annots = annots
        self.masks = masks
        self.sub_vol_size = sub_vol_size
        self.get_sub_tree = ExtRandomSubTree(0.0, 0.0, 0.0, seq_len, num_prev_pos)
        self.crop_and_pad = CropAndPad(sub_vol_size)
        self.convert_tree_to_targets = ConvertTreeToTargets(seq_len, num_prev_pos, sub_vol_size, class_dict)

    def __call__(self, input):
        annot_tree = self.annots[input['sample_id']]['branches'][input['tree_id']]
        selected_node = find_name(annot_tree, input['node_id'])
        _, past_traj_num = self.get_sub_tree.get_past_traj_start(selected_node, self.get_sub_tree.num_prev_pos)
        selected_node, past_traj_head_node = self.get_sub_tree.create_subtree_with_past_tr(selected_node,
                                                                                           past_traj_num, self.get_sub_tree.seq_len)
        data = self.convert_tree_to_targets.extract_sub_tree(selected_node)
        data['past_tr'] = self.convert_tree_to_targets.generate_past_trajectory_pp(selected_node,
                                                                                   data['root_position'])

        data.update({'index': input['sample_id'],
                     'tree_id': input['tree_id'],
                     'point_id': input['node_id'],
                     'point_type': input['point_type'],
                     'dist': input['distance']})

        image = self.crop_and_pad(self.images[input['sample_id']], selected_node.position, -1.0)
        if self.masks is not None:
            mask = self.crop_and_pad(self.masks[input['sample_id']], selected_node.position, 0.0)
        else:
            mask = None

        return data, image, mask


class LoadImageCropsAndTreesd(MapTransform):
    def __init__(self, keys, seq_len, num_prev_pos, sub_vol_size, class_dict,
                 mask, paths, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        images = {}
        annots = {}
        if mask:
            masks = {}
        else:
            masks = None

        image_reader = LoadImage(image_only=True)
        annot_reader = LoadAnnotPickle()
        self.norm = ScaleIntensity(minv=-1.0)

        for path in paths:
            index = [int(s) for s in re.findall(r'\d+', path[0])][-1]
            image = self.norm(image_reader(path[0]).unsqueeze(0))
            images[index] = image
            annots[index] = annot_reader(path[1])
            if mask:
                masks[index] = image_reader(path[2]).unsqueeze(0)

        self.mask = mask
        self.transform = LoadImageCropsAndTrees(seq_len, num_prev_pos, sub_vol_size,
                                                class_dict, images, annots, masks)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key], image, mask = self.transform(d[key])
            d['image'] = image
            d['mask'] = mask
        return d
