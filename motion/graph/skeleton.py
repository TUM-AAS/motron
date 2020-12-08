from typing import List

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

from common.quaternion import qrot, qmul
from motion.graph import Graph


class Skeleton(Graph):
    def __init__(self,
                 offsets: List,
                 parents: List,
                 joints_left: List,
                 joints_right: List,
                 state_representation: str):
        super().__init__(state_representation)
        self._offsets = torch.tensor(offsets)
        self._parents = torch.tensor(parents)
        self._joints_left = torch.tensor(joints_left)
        self._joints_right = torch.tensor(joints_right)
        self._compute_metadata()
        assert len(self._offsets.shape) == 2
        assert len(self._parents.shape) == 1
        assert self._offsets.shape[0] == self._parents.shape[0]

    @property
    def num_nodes(self):
        return len(self._parents)

    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix

    @property
    def chain_list(self):
        return self._chain_list

    def to_position(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_kinematics(x)

    def forward_kinematics(self, rotations: torch.Tensor, root_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        rotations = rotations.to('cpu')
        assert rotations.shape[-1] == 4
        batch_shape = list(rotations.shape[:-1])

        if root_positions is None:
            root_positions = torch.zeros((3,)).type_as(rotations)

        positions_world = []
        rotations_world = []

        if len(batch_shape) == 4:
            expanded_offsets = self._offsets.unsqueeze(dim=-2).expand(batch_shape + [self._offsets.shape[1]])
        else:
            expanded_offsets = self._offsets.expand(batch_shape + [self._offsets.shape[1]])

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions.expand(list(rotations[:, :, 0].shape[:-1]) + [3]))
                rotations_world.append(rotations[:, :, 0].contiguous())
            else:
                positions_world.append(qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i]) \
                                       + positions_world[self._parents[i]])
                if self._has_children[i]:
                    rotations_world.append(qmul(rotations_world[self._parents[i]], rotations[:, :, i].contiguous()))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=2)

    def visualize(self, x: torch.Tensor, fps=50):
        """
        Render or show an animation. The supported output modes are:
         -- 'interactive': display an interactive figure
                           (also works on notebooks if associated with %matplotlib inline)
         -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
         -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
         -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
        """
        x = x.detach().numpy()
        assert len(x.shape) == 3  # [T, N, 4]
        if x.shape[-1] == 4:
            x = self.to_position(x)
        assert x.shape[-1] == 3

        radius = np.max(self._offsets).item() * 5  # Heuristic that works well with many skeletons

        skeleton_parents = self._parents

        plt.ioff()
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(elev=20., azim=30)

        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5

        lines = []
        initialized = False

        trajectory = x[:, 0, [0, 2]]
        avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
        draw_offset = int(25 / avg_segment_length)
        spline_line, = ax.plot(*trajectory.T)
        camera_pos = trajectory
        height_offset = np.min(x[:, :, 1])  # Min height
        data = x.copy()
        data[:, :, 1] -= height_offset

        def update(frame):
            x = 0
            y = 1
            z = 2
            nonlocal initialized
            ax.set_xlim3d([-radius / 2 + camera_pos[frame, 0], radius / 2 + camera_pos[frame, 0]])
            ax.set_ylim3d([-radius / 2 + camera_pos[frame, 1], radius / 2 + camera_pos[frame, 1]])

            positions_world = data[frame]
            for i in range(positions_world.shape[0]):
                if skeleton_parents[i] == -1:
                    continue
                if not initialized:
                    col = 'red' if i in self._joints_right else 'black'  # As in audio cables :)
                    lines.append(ax.plot([positions_world[i, x], positions_world[skeleton_parents[i], x]],
                                         [positions_world[i, y], positions_world[skeleton_parents[i], y]],
                                         [positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y',
                                         c=col))
                else:
                    lines[i - 1][0].set_xdata([positions_world[i, x], positions_world[skeleton_parents[i], x]])
                    lines[i - 1][0].set_ydata([positions_world[i, y], positions_world[skeleton_parents[i], y]])
                    lines[i - 1][0].set_3d_properties([positions_world[i, z], positions_world[skeleton_parents[i], z]],
                                                      zdir='y')
            l = max(frame - draw_offset, 0)
            r = min(frame + draw_offset, trajectory.shape[0])
            spline_line.set_xdata(trajectory[l:r, 0])
            spline_line.set_ydata(np.zeros_like(trajectory[l:r, 0]))
            spline_line.set_3d_properties(trajectory[l:r, 1], zdir='y')
            initialized = True

        fig.tight_layout()
        anim = FuncAnimation(fig, update, frames=np.arange(0, data.shape[0]), interval=1000 / fps, repeat=False)
        plt.close()
        return anim

    def _compute_metadata(self):
        self._has_children = torch.zeros(len(self._parents), dtype=torch.bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._compute_adjacency_matrix()
        self._compute_chain_list()

    def _compute_adjacency_matrix(self):
        self._adjacency_matrix = torch.eye(len(self._parents))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self._parents[i] == j or self._parents[j] == i:
                    self._adjacency_matrix[i, j] = 0.5
                    self._adjacency_matrix[j, i] = 0.5

    def _compute_chain_list(self):
        self._chain_list = []
        for i, parent in enumerate(self._parents):
            if parent == -1:
                self._chain_list.append([i])
            else:
                self._chain_list.append(self._chain_list[parent] + [i])










