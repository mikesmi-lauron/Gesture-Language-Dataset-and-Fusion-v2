import numpy as np
from collections import Counter
class Scene:
    def __init__(self, scene):
        self.scene_id = scene["scene_id"]
        self.objects = scene["objects"]
        self.objs_coord = np.array([obj["position"] for obj in self.objects])
        self.objs_borders_coord = self.get_borders()
        self.commands = []
        self.feature_counts = self.calculate_feature_counts(self.objects)

    def get_borders(self):
        borders = []
        for row in self.objs_coord:
            border_points = (row + self.objs_coord) / 2
            borders.append(border_points)
        return np.array(borders)

    @staticmethod
    def calculate_feature_counts(objects):

        possible_features = set()
        for obj_tmp in objects:
            for key in obj_tmp["features"].keys():
                possible_features.add(key)

        """Count occurrences of features in the given objects."""
        feature_counts = {}
        for feature in possible_features:
            feature_counts[feature] = Counter(obj["features"][feature] for obj in objects)
        return feature_counts