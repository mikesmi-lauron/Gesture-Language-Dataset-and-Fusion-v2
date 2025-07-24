import numpy as np
import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import glob
import os
from data.Scene import Scene
from data.Command import Command




class MultiModalDataset(Dataset):
    def __init__(self, path2dataset="data/data_train/", modality_type="all",detection_type="aod",actions_complexity = 0):
        config_path = "data/data_config.json"  # Store for potential future use
        self.config = self.load_data_config(config_path) 
        self.config["detection_type"]= detection_type # Load config once
        self.config["modality_type"] = modality_type
        # Validate required keys and store values
        self.check_key("actions")
        self.update_config_actions(actions_complexity)
        self.actions_count = len(self.config["actions"].keys())
        self.check_key("object_properties")

        self.config["a2i"] = {action: i for i, action in enumerate(self.config["actions"])}
        self.config["op2i"] = {feature: i for i, feature in enumerate(self.config["object_properties"])}

        commands_pattern = path2dataset+ "*commands.csv"
        scenes_pattern = path2dataset + "*scenes.csv"
        self.commands = self.load_csv(commands_pattern, "commands")
        self.df_scenes = self.load_csv(scenes_pattern, "scenes")

        self.sceneid2idx = {}

        self.data_scenes = self.preprocess_scenes()
        self.data_commands = self.preprocess_commands()
        print("Dataset initialized")

    def update_config_actions(self,complexity):
        if complexity == 0:
            self.config["actions"] = dict(list(self.config["actions"].items())[:4] + [list(self.config["actions"].items())[-1]])
        elif complexity == 1:
            self.config["actions"] = dict(list(self.config["actions"].items())[:6] + [list(self.config["actions"].items())[-1]])
        elif complexity == "x":
            self.config["actions"]  =dict([list(self.config["actions"].items())[0]] + [list(self.config["actions"].items())[-1]])

    def preprocess_commands(self):
        all_commands = []
        for index, row in self.commands.iterrows():
            current_command = Command(index, row, self.config)

            #TODO think what is better? Loading scenes separately or adding them to commands
            """
            current_scene = self.df_scenes.loc[self.df_scenes["scene_id"] == current_command.scene_id].iloc[0]
            current_scene = Scene(index, current_scene, self.config)
            current_command.add_scene(current_scene)
            """
            all_commands.append(current_command)
        return all_commands

    def preprocess_scenes(self):
        all_scenes = []
        for index, row in self.df_scenes.iterrows():
            current_scene = Scene(index,row, self.config)
            self.sceneid2idx[current_scene.dataset_id] = index
            all_scenes.append(current_scene)
        return all_scenes

    def load_data_config(self, config_path):
        """Loads dataset configuration from a JSON file."""
        with open(config_path, 'r') as config_file:
            return json.load(config_file)

    def check_key(self, key):
        """Retrieves a required key from the config, raising an error if missing."""
        if key not in self.config:
            raise KeyError(f"Missing required key '{key}' in dataset configuration.")

    def load_csv(self, file_pattern, file_type):
        """Searches for files matching the pattern and loads the first one found, ensuring only one match."""
        matching_files = glob.glob(file_pattern)

        if len(matching_files) == 0:
            raise FileNotFoundError(f"No files found matching the pattern: '{file_pattern}'")
        elif len(matching_files) > 1:
            raise ValueError(f"Error: Multiple files found for '{file_type}': {matching_files}")

        # Load the first matching file
        try:
            return pd.read_csv(matching_files[0])
        except Exception as e:
            raise RuntimeError(f"Error loading CSV '{matching_files[0]}': {e}")


    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data_commands)

    def __getitem__(self, idx):
        """
        Retrieves the data sample and target for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing input features, target tensor, and the index.
        """
        command = self.data_commands[idx]
        scene = self.data_scenes[self.sceneid2idx[command.scene_id]]
        mask = torch.zeros(self.config["objects_limit"],dtype=torch.float32)
        mask[:scene.obj_count] = 1
        # Prepare input features
        input_features = [
            torch.tensor(command.ls_op[0], dtype=torch.float32),
            torch.tensor(command.g_pointing_obj_probs[0], dtype=torch.float32),
            torch.tensor(scene.spm, dtype=torch.float32),
            mask,
            torch.tensor(command.l_act_desc_vec[0], dtype=torch.float32),
            torch.tensor(command.g_act_desc_vec[0], dtype=torch.float32),
            torch.tensor(command.afm, dtype=torch.float32),
            torch.tensor(scene.sfm, dtype=torch.float32)
        ]

        # Handle modality-specific adjustments
        if self.config["modality_type"] == "language":
            input_features[1].fill_(float('nan'))
            input_features[5].fill_(0)
        elif self.config["modality_type"] == "gesture":
            input_features[0].fill_(float('nan'))
            input_features[4].fill_(0)

        # Process the target objects

        
        if self.config["detection_type"] == "aod":
            target_objects = command.preprocess_target_objects()
            target_actions = command.preprocess_target_actions()
            target = [
                torch.zeros((self.actions_count, self.config["objects_limit"]), dtype=torch.float32),
                torch.tensor(command.preprocess_target_objects(), dtype=torch.long).squeeze(),
                torch.tensor(command.preprocess_target_actions(), dtype=torch.long).squeeze()]
            target[0][target_actions, target_objects] = 1
        elif self.config["detection_type"] == "od":
            target = torch.tensor(command.preprocess_target_objects(), dtype=torch.long).squeeze()
        elif self.config["detection_type"] == "ad":
            target = torch.tensor(command.preprocess_target_actions(ad=True), dtype=torch.long).squeeze()
            input_features[6] = torch.tensor(command.afm[:-1,:], dtype=torch.float32)
        #else just continue with objects
        return input_features, target, idx

    def print_info(self,return_data = False):
        gesture_noise = []
        cmd_types = []

        for cmd in self.data_commands:
            cmd_types.append(cmd.type)
            gesture_noise.append(cmd.gpointing_noise)
        num_cmd_types = len(set(cmd_types))
        max_noise = max(max(gesture_noise), 5)
        noise_matrix = np.zeros((num_cmd_types, max_noise))
        for i, cmd in enumerate(self.data_commands):
            noise_matrix[cmd.type][cmd.gpointing_noise-1] += 1
        if return_data:
            return noise_matrix,set(cmd_types)
        else:
            not_unique = np.sum(noise_matrix[:,1:], axis=1)
            not_unique_total = np.sum(not_unique,axis=0)
            unique_total = np.sum(noise_matrix[:,0],axis=0)
            print(f"There is {unique_total} unique pointing gestures")
            print(f"There is {not_unique_total} pointing gestures that are not unique - more than 1 object has the same probability of selection")
            all_cmd = unique_total + not_unique_total
            print(f"Which means {round(not_unique_total/all_cmd *100,2)} % not unique and {round(unique_total/all_cmd *100,2)} % unique")
            print(noise_matrix)


if __name__ == '__main__':
    dataset = MultiModalDataset()
    dataset.print_info()
