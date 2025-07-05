import os
import re
import glob
import shutil
import zipfile
import pyvista
import pickle
import argparse
import numpy as np
import networkx as nx
from medpy.io import load
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
os.environ["BIGTREE_CONF_ASSERTIONS"] = ""
from bigtree import Node, find_name, levelordergroup_iter
from functools import lru_cache
import sys
sys.setrecursionlimit(10000)

import random
random.seed(37)


# extract all the files in the directory
def extract_files(directory):
    # get files with extension .zip
    zip_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.zip')]
    # get files with extension .tar.gz
    tar_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tar.gz')]

    # extract the zip files
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory)

    # extract the tar files
    for tar_file in tar_files:
        shutil.unpack_archive(tar_file, directory)


def rearrange_vtk_files(directory):
    vtk_dir = os.path.join(directory, "Synthetic Matthias LRZ Outputs")
    target_dir = os.path.join(directory, "vtk")
    os.makedirs(target_dir, exist_ok=True)
    folders = os.listdir(vtk_dir)
    custom_sort_key = lambda path: int(re.search(r'(\d+)', path).group(1))
    sorted_folders = sorted(folders, key=custom_sort_key)
    sorted_folders_paths = [os.path.join(vtk_dir, folder) for folder in sorted_folders]
    counter = 1
    for folder in sorted_folders_paths:
        # List all files in the current directory
        files = os.listdir(folder)
        sorted_files = sorted(files)
        # Iterate through each file
        for filename in sorted_files:
            # Construct the new file name
            new_filename = f"{counter}.vtk"
            src = os.path.join(folder, filename)
            dst = os.path.join(target_dir, new_filename)
            shutil.copyfile(src, dst)
            counter += 1


def fix_vtk_offset(directory):
    vtk_path = os.path.join(directory, "vtk")
    image_path = os.path.join(directory, "raw")
    target_dir = os.path.join(directory, "vtk_fixed")
    os.makedirs(target_dir, exist_ok=True)

    vtk_files = []
    img_files = []

    for file_ in os.listdir(vtk_path):
        file_ = file_[:-4]
        vtk_files.append(os.path.join(vtk_path, file_ + '.vtk'))
        img_files.append(os.path.join(image_path, file_ + '.nii.gz'))

    for idx, vtk_file in enumerate(vtk_files):
        vtk_data = pyvista.read(vtk_file)
        image_data, _ = load(img_files[idx])
        # correction of shift in the data (offset data is missing in the nifty file, so computing it manually)
        # get the center coordinates in x and y dim and add the correction offsets, for z dim correction is just 4.0
        shift = [np.shape(image_data)[0] / 2 - 1.8, np.shape(image_data)[1] / 2 + 8.3, 4.0]  # x,y,z shift
        coordinates = np.float32(np.asarray(vtk_data.points / 3.0 + shift))
        # lines = np.asarray(vtk_data.lines.reshape(vtk_data.n_cells, 3))

        vtk_data.points = coordinates
        vtk_data.save(os.path.join(target_dir, os.path.basename(vtk_file)))


def convert_vtk_to_networkx(mesh):
    # Get node coordinates from the mesh
    positions = np.float32(np.asarray(mesh.points))
    # center_offset = np.float32(np.asarray(mesh.center))[:2]
    # positions = rotate_xy_around_offset_z_axis(positions, 0, center_offset)
    positions_dict = dict(enumerate(positions))
    # Create node names
    node_names = np.arange(positions.shape[0])
    # Get edges from the mesh
    edges = mesh.lines.reshape(-1, 3)[:, 1:]
    edges = list(map(tuple, edges.tolist()))
    # Get the radius of the edges
    radii = np.asarray(mesh.cell_data['radius'])
    radii_dict = dict(zip(edges, radii))

    # Create networkx graph from the nodes, edges and radius
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    G.add_edges_from(edges)
    nx.set_node_attributes(G, positions_dict, 'position')
    nx.set_edge_attributes(G, radii_dict, 'radius')

    # min_radius = 6.0
    # # Remove edges with radius smaller than 3.0
    # edges_to_remove = [(u, v) for (u, v, data) in G.edges(data=True) if data['radius'] < min_radius]
    # G.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    # Extract connected components
    connected_components = list(nx.weakly_connected_components(G))

    # Create subgraphs from connected components
    subgraphs = [G.subgraph(component).copy() for component in connected_components]

    return subgraphs


# Reference: https://github.com/balzer82/3D-OccupancyGrid-Python/blob/master/Timings.py
def bresenham3D(startPoint, endPoint):
    path = []

    startPoint = [int(startPoint[0]), int(startPoint[1]), int(startPoint[2])]
    endPoint = [int(endPoint[0]), int(endPoint[1]), int(endPoint[2])]

    steepXY = (np.abs(endPoint[1] - startPoint[1]) > np.abs(endPoint[0] - startPoint[0]))
    if (steepXY):
        startPoint[0], startPoint[1] = startPoint[1], startPoint[0]
        endPoint[0], endPoint[1] = endPoint[1], endPoint[0]

    steepXZ = (np.abs(endPoint[2] - startPoint[2]) > np.abs(endPoint[0] - startPoint[0]))
    if (steepXZ):
        startPoint[0], startPoint[2] = startPoint[2], startPoint[0]
        endPoint[0], endPoint[2] = endPoint[2], endPoint[0]

    delta = [np.abs(endPoint[0] - startPoint[0]), np.abs(endPoint[1] - startPoint[1]),
             np.abs(endPoint[2] - startPoint[2])]

    errorXY = delta[0] / 2
    errorXZ = delta[0] / 2

    step = [
        -1 if startPoint[0] > endPoint[0] else 1,
        -1 if startPoint[1] > endPoint[1] else 1,
        -1 if startPoint[2] > endPoint[2] else 1
    ]

    y = startPoint[1]
    z = startPoint[2]

    for x in range(startPoint[0], endPoint[0], step[0]):
        point = [x, y, z]

        if (steepXZ):
            point[0], point[2] = point[2], point[0]
        if (steepXY):
            point[0], point[1] = point[1], point[0]

        errorXY -= delta[1]
        errorXZ -= delta[2]

        if (errorXY < 0):
            y += step[1]
            errorXY += delta[0]

        if (errorXZ < 0):
            z += step[2]
            errorXZ += delta[0]

        path.append(np.array(point))

    return path[1:]


def densify_networkx(graph):
    # check if graph is tree
    assert nx.is_tree(graph), "Graph is not a tree!"

    # Get all node positions from the graph
    node_positions = nx.get_node_attributes(graph, 'position')
    # Get all edges from the graph
    edges = list(graph.edges)
    # Get all edge radii from the graph
    edge_radii = nx.get_edge_attributes(graph, 'radius')

    # Go through each edge, and add nodes and edges in between
    for edge in edges:
        start_node = edge[0]
        end_node = edge[1]
        start_node_position = node_positions[start_node]
        end_node_position = node_positions[end_node]
        edge_radius = edge_radii[edge]

        # Calculate the inbetween node positions
        in_between_positions = bresenham3D(np.round(start_node_position), np.round(end_node_position))
        # computed by convex combination of end points
        mid_points = np.linspace(0, 1, 2 + len(in_between_positions))[1:-1]
        in_between_positions_cont = [(1 - x) * start_node_position + x * end_node_position for x in mid_points]

        # Remove the original edge
        graph.remove_edge(start_node, end_node)

        # Add the nodes in between
        for position in in_between_positions_cont:
            # Add the new node to the graph
            new_node = max(graph.nodes) + 1
            graph.add_node(new_node, position=position)
            # Add the new edge to the graph
            graph.add_edge(start_node, new_node, radius=edge_radius)
            start_node = new_node

        # connect last inbetween to the end node
        graph.add_edge(start_node, end_node, radius=edge_radius)

    # check if graph is tree
    assert nx.is_tree(graph), "Graph is not a tree!"
    return graph


@lru_cache(maxsize=None)
def convert_networkx_to_bigtree(graph):
    # Get all node positions from the graph
    node_positions = nx.get_node_attributes(graph, 'position')
    # Get all edge radii from the graph
    edge_radii = nx.get_edge_attributes(graph, 'radius')

    # Get the root point
    in_degrees = dict(graph.in_degree())
    root_node_id = [node for node, in_degree in in_degrees.items() if in_degree == 0]
    assert len(root_node_id) == 1, "Graph has multiple root nodes!"
    root_node_id = root_node_id[0]

    out_edges = list(graph.out_edges(root_node_id))
    # node radius is the maximum radius of the edges that start from that node
    node_radius = max([edge_radii[edge] for edge in out_edges])

    # Initialize the branch and point id for the root node
    branch_id = 0
    point_id = 0
    root_node_id_bt = str(branch_id) + "-" + str(point_id)
    node_pos = np.array(node_positions[root_node_id]).astype(float).tolist()
    root_node_bt = Node(str(root_node_id_bt), position=node_pos, radius=node_radius)
    branch_id_count = 0

    # create a dictionary to map the node id from networkx to bigtree
    nx_to_bt_node_id_dict = {root_node_id: root_node_id_bt}

    # Traverse the graph level by level using BFS and populate the bigtree nodes
    for i, level in tqdm(enumerate(nx.bfs_layers(graph, root_node_id)), position=1, desc="Levels",
                         leave=False, colour='blue', ncols=80):
        if i == 0:
            continue  # skip the root node
        for node in level:
            parent = list(graph.predecessors(node))
            assert len(parent) <= 1, "Graph has multiple parents!"
            parent = parent[0]
            node_radius = edge_radii[(parent, node)]
            parent_id_bt = nx_to_bt_node_id_dict[parent]
            branch_id = parent_id_bt.split('-')[0]
            point_id = parent_id_bt.split('-')[1]
            bifurcation = True if len(list(graph.out_edges(parent))) > 1 else False
            # if the node is a bifurcation, then create a new branch id by incrementing the branch id count and set the point id to 0
            if bifurcation:
                branch_id_count += 1
                branch_id = str(branch_id_count)
                point_id = '0'
            else:
                # if the node is not a bifurcation, then increment the point id
                point_id = str(int(point_id) + 1)
            node_id_bt = branch_id + "-" + point_id
            # Create a new node in bigtree
            node_pos = np.array(node_positions[node]).astype(float).tolist()
            Node(str(node_id_bt), position=node_pos, radius=node_radius,
                 parent=find_name(root_node_bt, parent_id_bt))
            # Add the node id to the mapping dictionary
            nx_to_bt_node_id_dict[node] = node_id_bt

    return root_node_bt


def convert_networkx_to_bigtree_wrapper(args):
    id, graph = args
    return id, convert_networkx_to_bigtree(graph)


def add_points_type_list(trees_with_metadata):
    """
    Add list of root point, bifurcation points, end points, and intermediate points
    """
    trees_with_metadata['all_ids'] = []
    trees_with_metadata['bifur_ids'] = []
    trees_with_metadata['endpts_ids'] = []
    trees_with_metadata['interm_ids'] = []
    trees_with_metadata['root_id'] = []
    trees_with_metadata['num_branches'] = []
    trees_with_metadata['num_points'] = []
    trees_with_metadata['branch_ids'] = []
    trees = trees_with_metadata['branches']
    for tree in trees:
        bifur_ids = []
        endpts_ids = []
        interm_ids = []
        all_ids = []
        for level in levelordergroup_iter(tree):
            for node in level:
                all_ids.append(node.node_name)
                if len(node.children) > 1:
                    bifur_ids.append(node.node_name)
                elif len(node.children) == 0:
                    endpts_ids.append(node.node_name)
                elif node.parent is None:
                    root_id = node.node_name
                else:
                    interm_ids.append(node.node_name)
        trees_with_metadata['all_ids'].append(all_ids)
        trees_with_metadata['bifur_ids'].append(bifur_ids)
        trees_with_metadata['endpts_ids'].append(endpts_ids)
        trees_with_metadata['interm_ids'].append(interm_ids)
        trees_with_metadata['root_id'].append(root_id)
        branch_ids = sorted(list(set([x.split('-')[0] for x in all_ids])))
        trees_with_metadata['branch_ids'].append(branch_ids)
        trees_with_metadata['num_branches'].append(len(branch_ids))

        # Count points for each branch
        branch_counts = {}
        for node in all_ids:
            branch_id = node.split('-')[0]
            if branch_id in branch_counts:
                branch_counts[branch_id] += 1
            else:
                branch_counts[branch_id] = 1
        # Create a list with counts for each corresponding branch in branch_ids
        num_points = [branch_counts[branch_id] for branch_id in branch_ids]
        trees_with_metadata['num_points'].append(num_points)

    return trees_with_metadata


def process_vtk_file(vtk_file):
    sample_id = re.search(r'(\d+)', vtk_file).group(1)
    tqdm.write("Converting Sample: " + sample_id)
    vtk_data = pyvista.read(vtk_file)

    # Convert the centerlines from vtk to networkx directed graphs
    graphs = convert_vtk_to_networkx(vtk_data)

    # Fill the inbetween nodes in the networkx directed graphs
    dense_graphs = [densify_networkx(graph) for graph in graphs]

    # Convert the networkx directed graphs to bigtree trees
    trees_with_metadata = {'branches': []}

    # Parallel conversion of networkx graphs to bigtree trees
    args_list = [(idx, graph) for idx, graph in enumerate(dense_graphs)]
    max_workers = 32  # Set this to the number of CPU cores you want to use
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(convert_networkx_to_bigtree_wrapper, args_list))

    # Create the bigtree_dicts from the results
    bigtree_dicts = {idx: result for idx, result in results}
    trees_with_metadata['branches'] = [bigtree_dicts[idx] for idx in range(len(bigtree_dicts))]

    # Add lists of root point, bifurcation points, end points, and intermediate points
    # tqdm.write("Adding Points Type List")
    trees_with_metadata = add_points_type_list(trees_with_metadata)

    # Create the bigtree_dicts from the results
    bigtree_dicts = {idx: result for idx, result in results}
    trees_with_metadata['branches'] = [bigtree_dicts[idx] for idx in range(len(bigtree_dicts))]

    # Save the bigtree trees
    bigtree_path = os.path.join(os.path.dirname(os.path.dirname(vtk_file)), "bigtree")
    with open(os.path.join(bigtree_path, sample_id + ".pickle"), 'wb') as handle:
        pickle.dump(trees_with_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_vtk_to_bigtree(directory):
    vtk_path = os.path.join(directory, "vtk_fixed")
    vtk_files = [os.path.join(vtk_path, file) for file in os.listdir(vtk_path)
                 if os.path.isfile(os.path.join(vtk_path, file))]
    bigtree_path = os.path.join(directory, "bigtree")
    os.makedirs(bigtree_path, exist_ok=True)

    # Define a custom sorting key using a lambda function
    custom_sort_key = lambda path: int(re.search(r'(\d+)', path).group(1))
    # Sort the paths using the custom sorting key
    sorted_vtk_files = sorted(vtk_files, key=custom_sort_key)

    # Use ProcessPoolExecutor to parallelize the outermost loop
    max_workers_outer = 32  # Set this to the number of CPU cores you want to use for the outer loop
    with ProcessPoolExecutor(max_workers=max_workers_outer) as executor_outer:
        executor_outer.map(process_vtk_file, sorted_vtk_files)


def convert_to_networkx(G, tree):
    """
    Function to traverse the bigtree tree and create the NetworkX graph
    """
    G.add_node(tree.node_name, position=tree.position, radius=tree.radius)

    if tree.parent is not None:
        G.add_edge(tree.parent.node_name, tree.node_name)

    for child in tree.children:
        convert_to_networkx(G, child)

    return G


def process_bigtree_file(bigtree_file):
    sample_id = re.search(r'(\d+)', bigtree_file).group(1)
    tqdm.write("Converting Sample: " + sample_id)
    with open(bigtree_file, 'rb') as handle:
        trees_with_metadata = pickle.load(handle)

    networkx_trees = []
    for idx, tree in enumerate(trees_with_metadata['branches']):
        G = nx.DiGraph()
        G = convert_to_networkx(G, tree)
        networkx_trees.append(G)

    trees_with_metadata['networkx'] = networkx_trees

    # Save the bigtree trees
    networkx_path = os.path.join(os.path.dirname(os.path.dirname(bigtree_file)), "networkx")
    with open(os.path.join(networkx_path, sample_id + ".pickle"), 'wb') as handle:
        pickle.dump(trees_with_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)


def bigtree_to_nx_graphs(directory):
    bigtree_path = os.path.join(directory, "bigtree")
    networkx_path = os.path.join(directory, "networkx")
    bigtree_files = [os.path.join(bigtree_path, file) for file in os.listdir(bigtree_path)
                     if os.path.isfile(os.path.join(bigtree_path, file))]
    bigtree_files = sorted(bigtree_files, key=lambda x: int(re.search(r'(\d+)', x).group(1)))
    os.makedirs(networkx_path, exist_ok=True)

    # Use ProcessPoolExecutor to parallelize the outermost loop
    max_workers_outer = 32  # Set this to the number of CPU cores you want to use for the outer loop
    with ProcessPoolExecutor(max_workers=max_workers_outer) as executor_outer:
        executor_outer.map(process_bigtree_file, bigtree_files)


def get_required_dir_struc(directory):
    # final dataset directory
    dst_dir = os.path.join(directory, "synthetic")
    paths_dict = {
        'annots_test': os.path.join(dst_dir, "annots_test"),
        'annots_train': os.path.join(dst_dir, "annots_train"),
        'annots_val': os.path.join(dst_dir, "annots_val"),
        'annots_val_sub_vol': os.path.join(dst_dir, "annots_val_sub_vol"),
        'images_test': os.path.join(dst_dir, "images_test"),
        'images_train': os.path.join(dst_dir, "images_train"),
        'images_val': os.path.join(dst_dir, "images_val"),
        'images_val_sub_vol': os.path.join(dst_dir, "images_val_sub_vol"),
        'masks_test': os.path.join(dst_dir, "masks_test"),
        'masks_train': os.path.join(dst_dir, "masks_train"),
        'masks_val': os.path.join(dst_dir, "masks_val"),
        'masks_val_sub_vol': os.path.join(dst_dir, "masks_val_sub_vol")}

    os.makedirs(dst_dir, exist_ok=True)
    for key in paths_dict:
        os.makedirs(paths_dict[key], exist_ok=True)

    img_dir = os.path.join(directory, "raw")
    mask_dir = os.path.join(directory, "seg")
    annot_dir = os.path.join(directory, "networkx")
    annot_files = sorted(os.listdir(annot_dir), key=lambda x: int(x.split('.')[0]))
    img_files = [os.path.join(img_dir, os.path.splitext(file)[0] + '.nii.gz') for file in annot_files]
    mask_files = [os.path.join(mask_dir, os.path.splitext(file)[0] + '.nii.gz') for file in annot_files]
    annot_files = [os.path.join(annot_dir, file) for file in annot_files]

    num_train_samples = 38
    num_val_samples = 2
    num_test_samples = 10
    val_trees = {0: [0, 12], 1: [1, 5]}

    # move the first num_train_samples samples to the training set
    start = 0
    end = start + num_train_samples
    for img_file, mask_file, annot_file in zip(img_files[start:end], mask_files[start:end], annot_files[start:end]):
        idx = str(int(re.compile(r'\d+').findall(img_file)[0]))
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(paths_dict['images_train'], idx + '.nii.gz'))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(paths_dict['masks_train'], idx + '.nii.gz'))
        shutil.copy(os.path.join(annot_dir, annot_file), os.path.join(paths_dict['annots_train'], idx + '.pickle'))

    # move the next num_val_samples samples to the sub volume (patch level) validation set
    start = num_train_samples
    end = start + num_val_samples
    for img_file, mask_file, annot_file in zip(img_files[start:end], mask_files[start:end], annot_files[start:end]):
        idx = str(int(re.compile(r'\d+').findall(img_file)[0]))
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(paths_dict['images_val_sub_vol'], idx + '.nii.gz'))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(paths_dict['masks_val_sub_vol'], idx + '.nii.gz'))
        shutil.copy(os.path.join(annot_dir, annot_file), os.path.join(paths_dict['annots_val_sub_vol'], idx + '.pickle'))

    # move specific trees from the num_val_samples samples to the validation set
    start = num_train_samples
    end = start + num_val_samples
    for f_id, (img_file, mask_file, annot_file) in enumerate(zip(img_files[start:end], mask_files[start:end], annot_files[start:end])):
        idx = str(int(re.compile(r'\d+').findall(img_file)[0]))
        for tree in val_trees[f_id]:
            shutil.copy(os.path.join(img_dir, img_file), os.path.join(paths_dict['images_val'], idx + str(tree) + '.nii.gz'))
            shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(paths_dict['masks_val'], idx + str(tree) + '.nii.gz'))
            annot_save_path = os.path.join(paths_dict['annots_val'], idx + str(tree) + '.pickle')
            shutil.copy(os.path.join(annot_dir, annot_file), annot_save_path)
            # Load pickle file
            with open(annot_save_path, 'rb') as handle:
                annot_data = pickle.load(handle)

            # remove the unwanted trees
            annot_filtered = {}
            for key in annot_data:
                annot_filtered[key] = [annot_data[key][tree]]

            # save the filtered annotation
            with open(annot_save_path, 'wb') as handle:
                pickle.dump(annot_filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # move the next num_test_samples samples to the test set
    start = num_train_samples + num_val_samples
    end = start + num_test_samples
    for img_file, mask_file, annot_file in zip(img_files[start:end], mask_files[start:end], annot_files[start:end]):
        idx = str(int(re.compile(r'\d+').findall(img_file)[0]))
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(paths_dict['images_test'], idx + '.nii.gz'))
        shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(paths_dict['masks_test'], idx + '.nii.gz'))
        shutil.copy(os.path.join(annot_dir, annot_file), os.path.join(paths_dict['annots_test'], idx + '.pickle'))


def create_sub_vol_eval_dataset(annot_path, bifur_prob, end_prob, root_prob,
                                seq_len, num_samples_per_annot, seq_range_half):
    if seq_range_half:
        seq_len = seq_len // 2
    annot_id = int(os.path.basename(annot_path).split('.')[0])

    annot_samples = []
    with open(annot_path, 'rb') as f:
        annot = pickle.load(f)

    num_bifur_samples = int(num_samples_per_annot * bifur_prob)
    num_end_samples = int(num_samples_per_annot * end_prob)
    num_root_samples = int(num_samples_per_annot * root_prob)
    num_rest_samples = num_samples_per_annot - num_bifur_samples - num_end_samples - num_root_samples

    # Get bifucation and end points
    types = ['bifur', 'end', 'root', 'interm']
    types_to_point_ids = {'bifur': 'bifur_ids', 'end': 'endpts_ids', 'root': 'root_id', 'interm': 'interm_ids'}
    num_type_samples = [num_bifur_samples, num_end_samples, num_root_samples, num_rest_samples]
    for t_i, type in enumerate(types):
        if type == 'root':
            type_points = [[id] for id in annot[types_to_point_ids[type]]]
        else:
            type_points = annot[types_to_point_ids[type]]
        total_type_points = sum([len(tr) for tr in type_points])

        # Cumulative index
        cumulative_index = -1
        index_mapping = {}
        for tree_id, lst in enumerate(type_points):
            for index_within_list, value in enumerate(lst):
                cumulative_index += 1
                index_mapping[cumulative_index] = [tree_id, index_within_list]

        # Generate num_type_samples random indices for type_points
        if type == 'root':
            type_samples_idx = random.choices(range(total_type_points), k=num_type_samples[t_i])
        else:
            type_samples_idx = random.sample(range(total_type_points), num_type_samples[t_i])

        for idx in type_samples_idx:
            tree_id, index_within_list = index_mapping[idx]
            node_name = type_points[tree_id][index_within_list]
            node_point_id = node_name.split('-')[1]
            while node_point_id == '0' and type == 'end':
                idx += 1
                tree_id, index_within_list = index_mapping[idx]
                node_name = type_points[tree_id][index_within_list]
                node_point_id = node_name.split('-')[1]
            if type == 'bifur':
                # sample a random index between (node_point_id - seq_len//2) and node_point_id
                index = random.randint(max(0, int(node_point_id) - seq_len), int(node_point_id))
            elif type == 'end':
                # sample a random index between (node_point_id - seq_len//2) and node_point_id - 1
                index = random.randint(max(0, int(node_point_id) - seq_len), int(node_point_id) - 1)
            elif type == 'root':
                branch_id = node_name.split('-')[0]
                branch_idx = annot['branch_ids'][tree_id].index(branch_id)
                branch_length = annot['num_points'][tree_id][branch_idx]
                # sample a random index between node_point_id and (node_point_id + seq_len//2)
                index = random.randint(int(node_point_id), min(branch_length - 1, int(node_point_id) + seq_len))
            else:
                index = int(node_point_id)
            distance = abs(index - int(node_point_id))
            new_node_name = node_name.split('-')[0] + '-' + str(index)
            assert new_node_name in annot['all_ids'][tree_id], 'new_node_name not in all_ids'

            # structure of annot_samples: [node_name, tree_id, annot_id, point_type, sample_point_id]
            annot_samples.append({'node_id': new_node_name, 'tree_id': tree_id, 'sample_id': annot_id,
                                  'point_type': type, 'distance': distance})

    return annot_samples


def get_annots_val_sub_vol(dataset_dir):
    bifur_prob = 0.2
    end_prob = 0.1
    root_prob = 0.1
    num_samples = 640
    seq_len = 10
    seq_range_half = False

    dataset_dir = os.path.join(dataset_dir, "synthetic")
    annot_dir = os.path.join(dataset_dir, 'annots_val_sub_vol')
    save_path = os.path.join(dataset_dir, 'annots_val_sub_vol.pickle')
    annot_paths = glob.glob(os.path.join(annot_dir, "*.pickle"))

    num_samples_per_annot = num_samples // len(annot_paths)
    samples_list = []
    for annot_path in annot_paths:
        samples_list += create_sub_vol_eval_dataset(annot_path, bifur_prob, end_prob, root_prob,
                                                    seq_len, num_samples_per_annot, seq_range_half)
    for sample in samples_list:
        print(sample)

    with open(save_path, 'wb') as f:
        pickle.dump(samples_list, f)


def main(dataset_dir):
    extract_files(dataset_dir)
    rearrange_vtk_files(dataset_dir)
    fix_vtk_offset(dataset_dir)
    convert_vtk_to_bigtree(dataset_dir)
    bigtree_to_nx_graphs(dataset_dir)
    get_required_dir_struc(dataset_dir)
    get_annots_val_sub_vol(dataset_dir)
    # TODO: check if all the required metadata has been computed and saved


if __name__ == '__main__':
    # Usage example:
    # python preprocess_synthetic_dataset.py /path/to/dataset
    parser = argparse.ArgumentParser(description="Process a dataset directory.")
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Path to the dataset directory'
    )
    args = parser.parse_args()
    main(args.dataset_dir)

