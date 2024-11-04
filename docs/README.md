### Annotation Details
The example_annotation.pickle file contains following fields:
- `branches`: List of vessel trees as bigtree objects. Each node of a vessel tree has the following attributes:
  - `name`: The name of a node is in `branch_id-point_id` format. The branch ids are integers starting from 0 for a single vessel tree. The point ids for each branch start from 0.
  - `label`: The label of the node. The label is 2 for a bifurcation point, 1 for an intermediate point, and 0 for an end point.
  - `position`: The position of the node in the image. The position is a list of [x, y, z] coordinates.
  - `radius`: The radius of the vessel at the node.
- `all_ids`: A list of lists of all the node names for each vessel tree.
- `bifur_ids`: A list of lists of all the bifurcation node names for each vessel tree.
- `endpts_ids`: A list of lists of all the end node names for each vessel tree.
- `interm_ids`: A list of lists of all the intermediate node names for each vessel tree.
- `root_id`: A list of root node names for each vessel tree.
- `num_branches`: The number of branches in each vessel tree.
- `branch_ids`: A list of lists of branch ids for each vessel tree.
- `num_points`: The number of points each branch in the branches in `branch_ids`.
- `networkx`: A list of networkx digraphs graphs for each vessel tree in `branches`.

### annots_val_sub_vol.pickle
The annots_val_sub_vol.pickle file contains the annotations required for performing patch-level evaluation. The file contains a list of dictionaries with following fields:
- `distance`: The distance of sampled point from the node `node_id`.
- `node_id`: The name of the node in `branch_id-point_id` format.
- `point_type`: The type of the selected point. `bifur`, `end`, `root` or `interm`.
- `sample_id`: The id of the volume sample.
- `tree_id`: The id of the tree in the `sample_id` volume.