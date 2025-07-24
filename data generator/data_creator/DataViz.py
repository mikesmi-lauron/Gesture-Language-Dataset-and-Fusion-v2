import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3D
import os
class DataViz:

    @staticmethod
    def check_and_create_folder(folder_path):
         # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    @staticmethod
    def line_intersects_bbox(P1, P2, bbox_min, bbox_max):
        # Initialize the t_min and t_max values for the intersection range
        t_min = -10.0
        t_max = 20.0

        # Check intersections with bounding box faces for each axis
        for axis in range(3):
            p1_axis = P1[axis]
            p2_axis = P2[axis]

            # If the line is parallel to the axis, skip it
            if p1_axis == p2_axis:
                continue

            # Compute the intersection parameter for the min and max of the current axis
            t_min_axis = (bbox_min[axis] - p1_axis) / (p2_axis - p1_axis)
            t_max_axis = (bbox_max[axis] - p1_axis) / (p2_axis - p1_axis)

            # Make sure t_min_axis is the smaller value and t_max_axis is the larger value
            if t_min_axis > t_max_axis:
                t_min_axis, t_max_axis = t_max_axis, t_min_axis

            # Update the global t_min and t_max for the intersection
            t_min = max(t_min, t_min_axis)
            t_max = min(t_max, t_max_axis)

            # If the valid range does not exist (no intersection), return an empty list
            if t_min > t_max:
                return []

        # Calculate the intersection points at t_min and t_max
        intersection_points = []

        # Calculate the point at t_min (entry point)
        point_min = P1 + t_min * (P2 - P1)
        intersection_points.append(point_min)

        # Calculate the point at t_max (exit point)
        point_max = P1 + t_max * (P2 - P1)

        intersection_points.append(point_max)

        return intersection_points

    @staticmethod
    def plot_3d_scene_with_views(scene, point1, point2, object_index=None, path2plot=None, sphere_color_mask=None,
                                 ax=None):

        bbox_min, bbox_max = np.array([-0.1, -0.1, -0.0]), np.array([0.6, 0.6, 0.5])
        views = {'Top view': (90, 0), 'Front view': (0, 0), 'Side view': (0, 90), 'Perspective view': (30, 30)}

        if ax is None:
            fig = plt.figure(figsize=(16, 10))
            for i, (title, (elev, azim)) in enumerate(views.items(), 1):
                ax = fig.add_subplot(2, 2, i, projection='3d')
                ax.set_title(title)
                DataViz.plot_objects(ax, scene.objects, bbox_min, bbox_max, object_index,
                                     sphere_color=sphere_color_mask)
                line_start, line_end = DataViz.get_line_intersection(point1, point2, bbox_min, bbox_max)
                DataViz.plot_line(ax, line_start, line_end, point1, point2)
                DataViz.set_axis_properties(ax, bbox_min, bbox_max, elev, azim)
            plt.tight_layout()
            if path2plot is None:
                plt.show()
            else:
                plt.savefig(path2plot)
                print(f"Plot saved as {path2plot}")
        else:
            DataViz.plot_objects(ax, scene.objects, bbox_min, bbox_max, object_index, sphere_color=sphere_color_mask)
            line_start, line_end = DataViz.get_line_intersection(point1, point2, bbox_min, bbox_max)

            DataViz.set_axis_properties(ax, bbox_min, bbox_max,elev = 30,azim=120)
            DataViz.plot_line(ax, line_start, line_end, point1, point2)
        return ax


    @staticmethod
    def plot_objects(ax, objects, bbox_min, bbox_max, object_index,sphere_color):
        """
        Plot objects (bounding boxes, spheres, etc.) in the 3D plot.
        """
        for idx, obj in enumerate(objects):
            sphere_colr = "gray"
            if sphere_color is not None and sphere_color[idx]:
                sphere_colr = "cyan"
            pos, size, color, shape = np.array(obj["position"]), np.array(obj["size"]) / 2, np.array(
                obj["color_id"]) / 255, obj["shape"]

            # Plot a transparent sphere around each object
            DataViz.plot_transparent_sphere(ax, pos, max(size)*1.9,sphere_colr)

            if shape == 'cube':
                DataViz.plot_cube(ax, pos, size, color)
            elif shape == 'sphere':
                DataViz.plot_sphere(ax, pos, size, color)
            elif shape == 'cylinder':
                DataViz.plot_cylinder(ax, pos, size, color)

    @staticmethod
    def plot_transparent_sphere(ax, pos, size,sphere_color):
        """
        Plot a transparent sphere around an object.
        """
        alpha = 0.3
        u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
        x = pos[0] + size * np.outer(np.cos(u), np.sin(v))
        y = pos[1] + size * np.outer(np.sin(u), np.sin(v))
        z = pos[2] + size * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=sphere_color, alpha=alpha)

    @staticmethod
    def plot_cube(ax, pos, size, color):
        """
        Plot a cube in the 3D plot.
        """
        box_vertices = [pos + np.array([dx, dy, dz]) * size for dx in [-1, 1] for dy in [-1, 1] for dz in [-1, 1]]
        faces = [
            [box_vertices[i] for i in [0, 1, 3, 2]],  # Bottom face
            [box_vertices[i] for i in [4, 5, 7, 6]],  # Top face
            [box_vertices[i] for i in [0, 1, 5, 4]],  # Front face
            [box_vertices[i] for i in [2, 3, 7, 6]],  # Back face
            [box_vertices[i] for i in [0, 2, 6, 4]],  # Left face
            [box_vertices[i] for i in [1, 3, 7, 5]]  # Right face
        ]
        ax.add_collection3d(Poly3DCollection(faces, color=color, linewidths=0.2, edgecolors='k', alpha=0.8))

    @staticmethod
    def plot_sphere(ax, pos, size, color):
        """
        Plot a sphere in the 3D plot.
        """
        u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
        x = pos[0] + size[0] * np.outer(np.cos(u), np.sin(v))
        y = pos[1] + size[1] * np.outer(np.sin(u), np.sin(v))
        z = pos[2] + size[2] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=0.8)

    @staticmethod
    def plot_cylinder(ax, pos, size, color):
        """
        Plot a cylinder in the 3D plot.
        """
        z = np.linspace(pos[2] - size[2], pos[2] + size[2], 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        X = size[0] * np.outer(np.cos(theta), np.ones_like(z))
        Y = size[0] * np.outer(np.sin(theta), np.ones_like(z))
        Z = np.outer(np.ones_like(theta), z)
        ax.plot_surface(X + pos[0], Y + pos[1], Z, color=color, alpha=0.8)

    @staticmethod
    def get_line_intersection(point1, point2, bbox_min, bbox_max):
        """
        Calculate intersection points of a line with a bounding box.
        """
        intersection_points = DataViz.line_intersects_bbox(np.array(point1), np.array(point2), bbox_min, bbox_max)
        if intersection_points == []:
            return point1, point2
        else:
            return intersection_points

    @staticmethod
    def plot_line(ax, line_start, line_end, point1, point2):
        """
        Plot the line connecting the two points and the markers at the start and end points.
        """
        ax.add_line(Line3D([line_start[0], line_end[0]], [line_start[1], line_end[1]], [line_start[2], line_end[2]],
                           color='green', linewidth=4))
        #ax.scatter(point1[0], point1[1], point1[2], color='cyan', marker='o', label="Start Point")
        #ax.scatter(point2[0], point2[1], point2[2], color='magenta', marker='o', label="End Point")

    @staticmethod
    def set_axis_properties(ax, bbox_min, bbox_max, elev, azim):
        """
        Set properties for the axis such as labels, limits, and view angles.
        """
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(bbox_min[0], bbox_max[0])
        ax.set_ylim(bbox_min[1], bbox_max[1])
        ax.set_zlim(bbox_min[2], bbox_max[2])
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True)

    @staticmethod
    def check_and_create_folder(folder_path):
        """
        Check if the folder exists, if not, create it.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)