import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from data_creator.DataViz import DataViz
from data_creator.DataCreator import DataCreator
import glob
from data_creator.Command import Command
from data_creator.Scene import Scene  # Assuming this exists
import os
import ast

class DatasetsInformation:
    def __init__(self):
        self.datasets = {}

    def add_dataset_name(self,dataset_name):
        self.datasets["dataset_name"] = {}

    def add_unique_commands_percentage(self,dataset_name,unique_commands_percentage):
        self.datasets[dataset_name]["unique_commands_percentage"] = unique_commands_percentage
class Dataset:
    def __init__(self, path2dataset):
        self.path2dataset = path2dataset
        type = ["train", "val", "test"]
        self.data = {}
        self.name = path2dataset.split("/")[-1]
        for t in type:
            self.data[t] = {"commands" : self.load_csv(f"{path2dataset}/{t}_commands.csv"),
                            "scenes" : self.load_csv(f"{path2dataset}/{t}_scenes.csv")}
            commands = self.data[t]["commands"]
            commands["g_pointing_obj_probs"] = commands['g_pointing_obj_probs'].apply(ast.literal_eval)
            commands["ones_count"] = commands['g_pointing_obj_probs'].apply(self.count_ones)
            self.data[t]["commands"] = commands


    def load_csv(self, path2file):
        return pd.read_csv(path2file)

    def count_ones(self,nested_list):
        """Counts the occurrences of 1 in the inner lists."""
        count = 0
        for inner_list in nested_list:
            count += inner_list.count(1)
        return count

    def get_unique_commands_percentage(self,type):
        commands =  self.data[type]["commands"]
        rows_with_more_than_one_one = len(commands[commands['ones_count'] > 1])
        return rows_with_more_than_one_one / len(commands) * 100


if __name__ == '__main__':

    name = "2025_04_30_17_01"
    sizes = ["small","medium","large"]
    s1 = [1,0.2,0.2]
    s2 = [10,50,250]
    scenes_paths = sorted(glob.glob(f'config/scenes_type*.json'))
    actions_paths = sorted(glob.glob(f'config/actions_complexity*.json'))
    datasets = []
    dataset_info = DatasetsInformation()
    for i,size in enumerate(sizes):
            for scene_path in scenes_paths:
                sc_type = scene_path.split("_")[-1].split(".")[0]
                dataset_name_1 = f"{size}_sc_{sc_type}"
                for actions_path in actions_paths:
                    ac_type = actions_path.split("_")[-1].split(".")[0]
                    dataset_tmp = Dataset(f"data/{name}/{dataset_name_1}_{ac_type}")
                    dataset_info.add_dataset_name(dataset_tmp.name)
                    dataset_info.add_unique_commands_percentage(dataset_tmp.name,dataset_tmp.get_unique_commands_percentage("train"))
                    print(dataset_tmp.get_unique_commands_percentage("train"))
    print("Dataset created")
