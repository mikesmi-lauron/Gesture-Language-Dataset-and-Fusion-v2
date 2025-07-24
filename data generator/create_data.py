from data_creator.DataCreator import DataCreator
from data_creator.SceneCreator import SceneCreator
from datetime import datetime
import random
import glob
import os
import json
def plot_commands(dataset):
    all_commands = [(scene, cmd) for scene in dataset.scenes for cmd in scene.commands]
    random_commands = random.sample(all_commands, 20)
    for scene, cmd in random_commands:
        cmd.plot_command(scene,dataset.config)


if __name__ == '__main__':

    time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    sizes = ["small"]
    purposes = ["train","val","test"]
    s1 = [1,0.2,0.2]
    s2 = [10,50,250]
    for i,size in enumerate(sizes):
        for j,purpose in enumerate(purposes):
            print(int(s1[j]*s2[i]))



            scenes_paths = sorted(glob.glob(f'config/scenes_type*.json'))
            scenes_paths = ["config/scenes_type_3simple.json"]
            actions_paths = sorted(glob.glob(f'config/actions_complexity*.json'))
            for scene_path in scenes_paths:
                sc_type = scene_path.split("_")[-1].split(".")[0]
                dataset_name_1 = f"{size}_sc_{sc_type}"
                path2scenes_with_pos = f'data/{time}/{dataset_name_1}/{purpose}_scenes_with_positions.json'
                scenes_creator = SceneCreator('config/config.json', int(s1[j] * s2[i]))
                scenes_creator.generate_scenes(scene_path, 'config/objects.json', path2scenes_with_pos)
                for actions_path in actions_paths:
                    ac_type = actions_path.split("_")[-1].split(".")[0]
                    dataset = DataCreator("config/config.json",path2scenes_with_pos,actions_path)
                    dataset.create_dataset()
                    dataset.save_to_csv(f"data/{time}/{dataset_name_1}_{ac_type}/{purpose}")
                   # dataset.save_to_json(f"data/{time}/{dataset_name}/scenes_type_{type}/{dataset_name}_{type}_all_data.json")
                    #if purpose == "eval" and size == "small":
                        #plot_commands(dataset)
                    print(f"Dataset {dataset_name_1}_{ac_type} created")
    print("Dataset created")
