import bpy
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
from util import *
from copy import deepcopy
import sys

sys.setrecursionlimit(10000)

class Branch:
    def __init__(self):
        self.parent = None
        self.children = []

        self.local_cs = np.eye(3)

        # Start and end points
        self.start_point = None
        self.end_point = None

        # Scalar parameters
        self.length = 0
        self.x_ang = 0
        self.z_ang = 0
        self.cs_area = 0
        self.dry = False


class Tree:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.base = None

        self.graph_points = np.empty((3, 0), dtype=float)
        self.graph_edges = np.empty((2, 0), dtype=int)
        self.graph_edge_metadata = []
    
    def create_tree_graph(self, params: dict):
        state = {
            'level': 0,
            'cs_area': params["cs_area_base"],
            'branch_len': params["branch_len_base"],
            'declination': 0.0,
            'break_prob': params['init_break_prob'],
            'dry': False
        }

        self.base = create_branch_recursive(state, params, None)
    
    def gather_points(self):
        if self.base is None:
            print("Tree not created yet")
            return

        self.graph_points = np.append(self.graph_points, self.base.start_point, axis=1)
        self.gather_points_recursive(self.base, 0)

    def gather_points_recursive(self, branch: Branch, parent_end_idx: int):
        self.graph_edges = np.append(self.graph_edges, np.array([[parent_end_idx], [self.graph_points.shape[1]]]), axis=1)
        
        metadata = {
            'cs_area': branch.cs_area,
            'dry': branch.dry
        }
        self.graph_edge_metadata.append(metadata)
        end_idx = self.graph_points.shape[1]
        self.graph_points = np.append(self.graph_points, branch.end_point, axis=1)
        
        for c in branch.children:
            self.gather_points_recursive(c, end_idx)

    def draw_tree(self):
        """
        Draw the tree graph using Matplotlib
        """

        if self.base is None:
            print("Tree not created yet")
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        pnts = self.graph_points

        for i in range(self.graph_edges.shape[1]):
            start_idx = self.graph_edges[0, i]
            end_idx = self.graph_edges[1, i]

            if self.graph_edge_metadata[i]['dry']:
                color = 'r'
            else:
                color = 'k'

            # if i == self.graph_edges.shape[1]-1:
            #     color = 'g'

            ax.plot(
                [pnts[0, start_idx], pnts[0, end_idx]],
                [pnts[1, start_idx], pnts[1, end_idx]],
                [pnts[2, start_idx], pnts[2, end_idx]],
                color=color, linewidth=1, marker='o', markersize=2
            )
        
        set_axes_equal(ax)

        plt.show()
 
parser = argparse.ArgumentParser()
parser.add_argument('--tree_type', type=str, default='A', help='Type of tree to generate')

def main(args):
    # Delete all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    branch_factor = 4 # Average number of branches at branch point
    branch_factor_stddev = 0.5 # Average number of branches at branch point
    branch_len_base = 2 # Average length of trunk before the first branching
    branch_len_factor = 0.75 # Shortening of average branch length for each recursion level
    branch_len_stddev_mult = 0.1 # Standard deviation of the branch length
    cs_area_base = 0.25 # Cross-sectional area in square meters at the base of the trunk
    cs_area_stddev_mult = 0.1 # Multiplier used to compute the standard deviation of the cross-sectional area from the actual area
    cs_area_min = 1e-4
    recursion_levels = 4
    recursion_levels_stddev = 0.0
    split_angle = 20 # Average angle (degrees) between branches around X axis
    split_angle_stddev = 15 # Standard deviation of the angle between branches around X axis

    cut_prob = 0.0
    break_prob = 0.0
    dry_prob = 0.0
    dry_break_mult = 4 # Multiplier of break probability if dry

    params = {
        'branch_factor': branch_factor,
        'branch_factor_stddev': branch_factor_stddev,
        'branch_len_base': branch_len_base,
        'branch_len_factor': branch_len_factor,
        'branch_len_stddev_mult': branch_len_stddev_mult,
        'cs_area_base': cs_area_base,
        'cs_area_stddev_mult': cs_area_stddev_mult,
        'cs_area_min': cs_area_min,
        'cut_prob': cut_prob,
        'init_break_prob': break_prob,
        'dry_prob': dry_prob,
        'dry_break_mult': dry_break_mult,
        'recursion_levels': recursion_levels,
        'recursion_levels_stddev': recursion_levels_stddev,
        'split_angle': split_angle,
        'split_angle_stddev': split_angle_stddev
    }

    tree = Tree(params)
    tree.create_tree_graph(params)
    tree.gather_points()
    tree.draw_tree()

def create_branch_recursive(state:dict, params: dict, parent: Branch):
    """
    Step of the recursive tree creation algorithm

    Parameters
    ----------
    cs_area : float
        Cross-sectional area of the branch
    
    Returns
    -------
    bpy.types.Object
        The created branch
    """

    branch_len_avg = params['branch_len_base'] * (params['branch_len_factor'] ** state['level'])
    branch_len_stddev = branch_len_avg * params['branch_len_stddev_mult']
    branch_len = np.random.normal(branch_len_avg, branch_len_stddev)

    branch_terminate = np.random.normal(params['recursion_levels'], params['recursion_levels_stddev']) < state['level']
    branch_break = np.random.uniform() < state["break_prob"]
    branch_cut = np.random.uniform() < params["cut_prob"]
    if parent.dry:
        branch_dry = True
    else:
        branch_dry = np.random.uniform() < params["dry_prob"]

    if branch_break:
        break_len = 0.2 # TODO: add usable fnc.
    
    if branch_cut:
        cut_len = 0.05 # TODO: add usable fnc.
    
    if state['dry']:
        state["break_prob"] *= params['dry_break_mult']
    elif branch_dry:
        dry_len = 0.1 # TODO: add usable fnc.
        state['dry'] = True
        state["break_prob"] *= params['dry_break_mult']

    branch = Branch()
    branch.parent = parent

    if branch_break and branch_cut:
        if break_len < cut_len:
            branch_len = break_len
        else:
            branch_len = cut_len
    elif branch_break:
        branch_len = break_len
    elif branch_cut:
        branch_len = cut_len

    branch.length = branch_len

    if parent is None:
        branch.x_ang = np.random.normal(0.0, params['split_angle_stddev'])
        branch.z_ang = 0.0
        branch.start_point = np.zeros((3, 1))
        branch.local_cs = np.eye(3)
    else:
        branch.x_ang = np.random.normal(
            params['split_angle'], params['split_angle_stddev'])
        state['declination'] += branch.x_ang
        branch.start_point = parent.end_point

        if state['level'] == 1:
            branch.z_ang = np.random.uniform(0.0, 360.0)
        else:
            branch.z_ang = gen_z_ang(state['declination'])
            # branch.z_ang = 0.0
        
        rotmat = get_rotmat_xz(branch.x_ang, branch.z_ang)
        branch.local_cs = parent.local_cs @ rotmat

    branch.end_point = branch.start_point + branch.local_cs @ np.array([[0],[0],[branch.length]])    

    if not (branch_terminate) and not (branch_break) and not (branch_cut):
        state['level'] += 1
        branch_num = math.floor(np.random.normal(params['branch_factor'], params['branch_factor_stddev']))

        cs_area_stddev = params['cs_area_stddev_mult']*state['cs_area']
        cs_area = np.random.normal(state['cs_area'], cs_area_stddev, branch_num)
        cs_area = norm_vec(cs_area)

        for b in range(branch_num):
            child_state = deepcopy(state)
            child_state['cs_area'] = cs_area[b]
            child_branch = create_branch_recursive(child_state, params, branch)
            
            branch.children.append(child_branch)

    return branch


def create_unit_bezier_curve(name: str, loc: np.ndarray, ori: np.ndarray) -> bpy.types.Object:
    """
    Create a unit bezier curve

    Creates a unit bezier curve with two points, one at the origin and one at 
    the origin + [0, 0, 1].

    Parameters
    ----------
    name : str
        Name of the curve
    loc : np.ndarray, optional
        Location of the curve (default is [0, 0, 0])
    ori : np.ndarray, optional
        Orientation of the curve (default is [0, 0, 1])

    Returns
    -------
    bpy.types.Object
        The created curve
    """

    bpy.ops.curve.primitive_bezier_curve_add(radius=1.0, enter_editmode=False, align='WORLD')
    curve = bpy.context.object
    curve.name = name

    ori = norm_vec(ori)

    # Move curve points to create unit line along Z-axis
    curve.data.splines[0].bezier_points[0].co = loc
    curve.data.splines[0].bezier_points[1].co = loc + ori

    # Move curve handles to create a straight line
    curve.data.splines[0].bezier_points[0].handle_left = loc - ori
    curve.data.splines[0].bezier_points[0].handle_right = loc + ori
    curve.data.splines[0].bezier_points[1].handle_left = loc + 2*ori
    curve.data.splines[0].bezier_points[1].handle_right = loc

    return curve


def gen_z_ang(declination: float) -> float:
    """
    Generate a random angle around the branch Z-axis based on Weber-Penn model

    Parameters
    ----------
    declination : float
        Angle around the branch X-axis relative to the whole tree Z-axis

    Returns
    -------
    float
        Random angle around the branch Z-axis
    """

    side = np.random.choice([-1, 1])
    angle = side*(20 + 0.75*(30 + abs(declination - 90))*np.random.uniform(0,1))

    return angle


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([0.0, 2*plot_radius])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)