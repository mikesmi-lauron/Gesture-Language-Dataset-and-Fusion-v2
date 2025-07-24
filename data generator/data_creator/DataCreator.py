import json
from collections import Counter
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from data_creator.Scene import Scene
from data_creator.Command import Command
import os
import pandas as pd

PLOT_PG = False
# Define the fixed action for command generation
ACTIONS = ["pick up"]
GESTURES = ["pick up", "pointing"]


def load_json(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, 'r') as file:
        return json.load(file)


class DataCreator:
    def __init__(self, path2config_file="config.json", path2scenes=None, path2actions=None):
        self.scenes = None
        self.actions_structures = self.load_actions_structures(path2actions)
        self.path2scenes = path2scenes
        self.commands = []
        self.config = load_json(path2config_file)
        self.command_structures = self.config["command_structure"]
        self.epsilon = 1  #radius is 1.1 times the size of the object
        self.bsphere_radius = 1.9

    def load_actions_structures(self, path2actions):
        if path2actions is None:
            actions_structures = [{
                "id": 0,
                "actions": ["push"],
                "language": "move",
                "gesture": "g0"
            }]
        else:
            actions_structures = load_json(path2actions)
        return actions_structures
    def create_dataset(self):
        """
        Processes each scene and generates commands for objects based on predefined structures.
        Args:
            path2scenes_file (str): Path to the JSON file containing scene data.
        Returns:
        """

        #create scenes
        data = load_json(self.path2scenes)
        self.scenes = [Scene(scene) for scene in data["Scenes"]]

        for i, scene in enumerate(self.scenes):
            cmd_in_scene_counter = 0
            for obj in scene.objects:
                for j, command_structure in enumerate(self.command_structures):
                    for action in self.actions_structures:
                        #if target is single
                        target_objs = [obj]

                        cmd_structure = command_structure["structure"]
                        cmd_type = self.config["command_structure"][j]["type"]

                        temp_command = Command(target_objs, cmd_structure, cmd_type, scene, cmd_in_scene_counter)
                        temp_command.update_l_obj_props()
                        temp_command.update_l_words()
                        temp_command.update_action(action, self.config)
                        self.calculate_distance_probability(temp_command, scene)
                        self.commands.append(temp_command)

                        cmd_in_scene_counter += 1

    @staticmethod
    def distance_from_line_to_point(point1, point2, point3):
        # Convert points to numpy arrays
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)

        # Vector from p1 to p2 (direction of the line)
        line_vec = p2 - p1
        # Vector from p1 to p3
        point_vec = p3 - p1

        # Projection of point_vec onto line_vec
        proj_length = np.dot(point_vec, line_vec) / np.linalg.norm(line_vec)
        proj_vec = (proj_length / np.linalg.norm(line_vec)) * line_vec

        # Vector perpendicular from point3 to the line
        perp_vec = point_vec - proj_vec

        # Distance is the magnitude of the perpendicular vector
        distance = np.linalg.norm(perp_vec)

        return distance

    def generate_random_gpointing(self, target_obj):
        radius = (max(target_obj['size']) / 2) * self.epsilon
        distance = random.uniform(0, radius)

        # Generate random angles for spherical coordinates
        theta = random.uniform(0, 2 * math.pi)  # Azimuthal angle (0 to 2*pi)
        phi = random.uniform(0, math.pi)  # Polar angle (0 to pi)

        # Convert spherical coordinates to Cartesian coordinates
        x_offset = distance * math.sin(phi) * math.cos(theta)
        y_offset = distance * math.sin(phi) * math.sin(theta)
        z_offset = distance * math.cos(phi)
        x1, y1, z1 = target_obj["position"]
        # Calculate new coordinates relative to the object position
        new_gpoint = [x_offset + x1, y_offset + y1, z_offset + z1]
        return new_gpoint

    def calculate_distance_probability(self, command, scene):
        pointing_gest_coords = []
        for target_obj in command.target_objs:
            pointing_gest_coords.append(self.generate_random_gpointing(target_obj))
        command.update_gpointing_coord(pointing_gest_coords)

        # compute bouding spheres for the whole objects in the scenes
        bspheres = [max(scene_obj['size']) / 2 * self.bsphere_radius for scene_obj in scene.objects]

        # Initialize variables for storing distances and intersection states
        distances = []
        intersection = []

        g_obj_probs = []
        for pointing_gest_coord_i in command.g_pointing_coords:
            for i, obj_tmp in enumerate(scene.objects):
                obj_coord = obj_tmp["position"]

                # Calculate the distance from the line (hand position) to the object's coordinate
                obj_dist = self.distance_from_line_to_point(self.config["hand_position"],
                                                            pointing_gest_coord_i, obj_coord)

                # Adjust distance by subtracting the bounding sphere radius
                obj_dist -= bspheres[i]

                # Check if the object is within the bounding sphere
                if obj_dist >= 0:
                    distances.append(obj_dist)
                    intersection.append(False)
                else:
                    intersection.append(True)

            # Calculate inverse distances and initialize probabilities
            inverse_distances = 1 / np.array(distances)
            probabilities = np.ones(len(intersection))

            # Adjust probabilities based on intersection state and distances
            probabilities[~np.array(intersection)] = inverse_distances / np.sum(inverse_distances)
            probabilities = probabilities.tolist()

            g_obj_probs.append(probabilities)
        command.update_g_obj_probs(g_obj_probs)

    def convert_booleans(self,obj):
        """Recursive function to convert booleans in nested objects to 'true'/'false' strings."""
        if isinstance(obj, dict):
            return {key: self.convert_booleans(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_booleans(item) for item in obj]
        elif isinstance(obj, bool):
            return 'true' if obj else 'false'
        else:
            return obj

    def save_to_csv(self, file_path):
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        cmds = []
        scenes = []
        for scene in self.scenes:
            tmp_scene = {
                "scene_id": scene.scene_id,
                "objects": scene.objects,
            }
            tmp_scene = self.convert_booleans(tmp_scene)
            scenes.append(tmp_scene)
        df = pd.DataFrame(scenes)
        df.to_csv(file_path + "_scenes.csv", index=False)

        for cmd in self.commands:
            tmp_cmd = cmd.to_json()
            cmds.append(tmp_cmd)
        df = pd.DataFrame(cmds)
        df.to_csv(file_path + "_commands.csv", index=False)

