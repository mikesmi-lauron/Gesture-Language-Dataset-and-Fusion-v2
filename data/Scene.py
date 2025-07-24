import numpy as np
import json
class Scene:
    def __init__(self, scene_id,scene_info, config_info):
        """
        Initializes the Scene object with the scene ID, scene information,
        and configuration, and generates the Scene Properties Matrix (SPM).

        Args:
            scene_id: (int) The ID of the scene when loaded from the dataset.
            scene_dataset_id (str): The unique ID of the scene in the format 'XX-YY'.
            scene_type (int): The type of the general scene.
            scene_objects (list): List of scene objects, each containing features.
            config_info (dict): Configuration of the dataset.
        """

        scene_objects = scene_info["objects"].replace("'", '"')
        self.dataset_config = config_info
        self.dataset_id = scene_info["scene_id"]
        self.objects = json.loads(scene_objects)

        self.obj_count = len(self.objects)
        self.id = scene_id
        self.type = int(self.dataset_id.split('-')[0])
        
        self.spm = self.compute_spm()
        self.sfm = self.create_sfm()
    
    def create_sfm(self):
        actions_info = self.dataset_config["actions"]
        actions_tags = actions_info.keys()
        actions_features = set()
        for tag in actions_tags:
            actions_features.update(actions_info[tag]["conditions"].keys())
        sfm = np.zeros((len(actions_features), self.dataset_config["objects_limit"]))
        for i, feature in enumerate(actions_features):
            for j, obj in enumerate(self.objects):
                if feature in obj["features"]:
                    if obj["features"][feature] == 'true':
                        sfm[i, j] = 1
                    if obj["features"][feature] == 'false':
                        sfm[i, j] = -1
        return sfm

    def compute_spm(self):
        """
        Generates the Scene Properties Matrix (SPM) based on scene features
        and object properties from the config.

        Args:
            config_info (dict): Configuration containing object properties and max object count.

        Returns:
            np.ndarray: Scene Properties Matrix (SPM), where rows represent object properties
                        and columns represent scene objects.
        """
        feature_count = len(self.dataset_config["object_properties"])
        max_obj_count = self.dataset_config["objects_limit"]
        spm = np.zeros((feature_count, max_obj_count))
        for i, obj in enumerate(self.objects):
            # Get features
            obj_features = [feature for feature in obj["properties"].values()]

            # Set object properties to 1 if feature is in the object's features
            for j, general_feature in enumerate(self.dataset_config["object_properties"]):
                if general_feature in obj_features:
                    spm[j, i] = 1

        return spm

    def get_possible_targets_idx(self,primary_object_idx):
        column = self.spm[:,primary_object_idx]
        return np.where((self.spm.T == column).all(axis=1))[0].tolist()