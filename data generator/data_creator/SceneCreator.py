import json
import random
import os

PROB_STANDING_ON_OBJ = 0.3
MAX_TOWER_HEIGHT = 3
PROB_FREE = 0.25

class SceneCreator:
    """Generates scenes with objects placed in a working space."""

    def __init__(self, config_file, samples_per_scene=None):
        """Initialize with config settings."""
        with open(config_file, 'r') as file:
            self.config = json.load(file)
        if samples_per_scene is not None:
            self.config["scene_variation_count"] = samples_per_scene

    @staticmethod
    def is_overlapping(new_position, existing_positions, object_size, objects_sizes):
        """Check if a position overlaps with existing objects."""
        min_distance = max(object_size) / 2
        safe_dist = 0.02
        return any(
            abs(new_position[0] - pos[0]) < (min_distance + objects_sizes[i] / 2 + safe_dist) and
            abs(new_position[1] - pos[1]) < (min_distance + objects_sizes[i] / 2 + safe_dist)
            for i, pos in enumerate(existing_positions)
        )

    def generate_scene_objects(self, object_data, object_count_map, scene_id):
        """Create objects for a scene."""
        scene_objects = []
        objects_sizes = []
        height_level = []
        occupied = []
        existing_positions = []
        counter = 0
        bbox = self.config["working_space_bbox"]

        objects_ids = [item[0] for item in object_count_map.items() for _ in
                       range(item[1])]  # Convert to list of tuples
        random.shuffle(objects_ids)

        for obj_id in objects_ids:
            obj = next((o for o in object_data if o["id"] == obj_id), None)
            if not obj:
                raise Exception("Obj ID not found")

            position = []
            if random.random() < PROB_STANDING_ON_OBJ and len(existing_positions) > 0:
                indexes = list(range(len(existing_positions)))  # Create a list of indexes
                random.shuffle(indexes)

                for index in indexes:
                    tmp_position = existing_positions[index]
                    potential_height = height_level[index] + 1
                    if (objects_sizes[index] >= obj["size"][0] and occupied[index] == False
                            and scene_objects[index]["shape"] != "sphere" and potential_height < MAX_TOWER_HEIGHT):
                        position = tmp_position.copy()
                        position[2] = obj["size"][-1] / 2 + objects_sizes[index] / 2 + position[2]
                        occupied[index] = True
                        scene_objects[index]["features"]["liftable"] = False
                        occupied.append(False)
                        height_level.append(potential_height)
                        break
            if len(position) <= 0:

                size_idx = 1
                if obj["shape"] == "sphere":
                    size_idx = 0
                position = [
                    random.uniform(bbox["x_min"] + obj["size"][0] / 2, bbox["x_max"] - obj["size"][0] / 2),
                    random.uniform(bbox["y_min"] + obj["size"][size_idx] / 2,
                                   bbox["y_max"] - obj["size"][size_idx] / 2),
                    obj["size"][-1] / 2
                ]

                attempts = 0
                max_attempts = 1000

                while self.is_overlapping(position, existing_positions, obj["size"], objects_sizes):
                    if attempts >= max_attempts:
                        print("have to regenerate")
                        return None

                    position[:2] = [
                        random.uniform(bbox["x_min"], bbox["x_max"]),
                        random.uniform(bbox["y_min"], bbox["y_max"])
                    ]

                    attempts += 1
                occupied.append(False)
                height_level.append(0)
            objects_sizes.append(max(obj["size"]))
            existing_positions.append(position)

            if obj["shape"] == "sphere":
                movability = False
            else:
                movability = True

            if random.random() < PROB_FREE:
                obj_is_free = False
            else:
                obj_is_free = True

            scene_objects.append({
                "id": f"{scene_id}-{counter:02}-{obj['id']}",
                "shape": obj["shape"],
                "color": obj["color"],
                "color_id": obj["color_id"],
                "size": obj["size"],
                "position": position,
                "properties": obj["properties"],
                "features":
                    {
                        "liftable": True,
                        "movable": movability,
                        "free": obj_is_free
                    }
            })
            counter += 1

        return scene_objects

    def generate_scenes(self, input_scenes_file, input_objects_file, output_file):
        """Generate scenes and save them to a file."""
        with open(input_scenes_file, 'r') as file:
            scenes_data = json.load(file)
        with open(input_objects_file, 'r') as file:
            objects_data = json.load(file)

        scenes = []
        for scene in scenes_data['Scenes']:
            scene_id = scene['id']
            object_count_map = scene['objects']
            for i in range(self.config["scene_variation_count"]):
                formatted_id = f"{scene_id:02}-{i:02}"
                objects = None
                while objects is None:
                    objects = self.generate_scene_objects(objects_data['objects'], object_count_map, formatted_id)
                scenes.append({"scene_id": formatted_id, "objects": objects})

        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'w') as file:
            json.dump({"Scenes": scenes}, file, indent=4)


# Usage
if __name__ == '__main__':
    creator = SceneCreator('config.json')
    creator.generate_scenes('scenes.json', 'objects.json', 'scenes_with_positions.json')
