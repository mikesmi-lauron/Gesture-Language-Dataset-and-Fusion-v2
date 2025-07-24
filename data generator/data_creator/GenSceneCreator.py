import random
import json
import datetime


class GenSceneCreator:
    def __init__(self, config, objects):
        self.config = config
        self.objects = objects

    def generate_scene(self, scene_id):
        scene = {}
        scene['id'] = scene_id
        scene['objects'] = {}

        object_count = random.randint(2, 5)  # Number of unique objects in this scene

        # Randomly select objects and their quantities
        selected_objects = random.sample(self.objects, object_count)

        # Make sure there is at least one pair of the same object
        object_pair = random.choice(selected_objects)
        scene['objects'][object_pair['id']] = 2  # Adding pair of the same object

        # Add random number of the other objects
        for obj in selected_objects:
            if obj != object_pair:
                scene['objects'][obj['id']] = random.randint(1, 2)

        return scene

    def generate_multiple_scenes(self):
        scenes = []
        for i in range(1, self.config["scene_variation_count"] + 1):
            scene = self.generate_scene(i)
            scenes.append(scene)
        return scenes


def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def save_scenes_to_file(scenes):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"config/general_scenes_{timestamp}.json"

    # Save the scenes to a JSON file
    with open(filename, 'w') as file:
        json.dump({"Scenes": scenes}, file, indent=2)


def main():
    # Load the configuration and objects from their respective JSON files
    config = load_json('config/config.json')["config"]  # Load the config.json file
    objects = load_json('config/objects.json')["objects"]  # Load the objects.json file

    # Create an instance of GenSceneCreator
    scene_creator = GenSceneCreator(config, objects)

    # Generate multiple scenes based on the config
    generated_scenes = scene_creator.generate_multiple_scenes()

    # Save the generated scenes to a file
    save_scenes_to_file(generated_scenes)

    # Optionally, print the generated scenes
    print(f"Scenes saved as 'general_scenes_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json'")


if __name__ == "__main__":
    main()
