from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import json
import time
import os
from PIL import Image
from datetime import datetime

# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.getObject('sim')

# Function to read the scene from a JSON file
def load_scene_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to create an object in the simulation
def create_object(object_data):
    # Map shape types
    shape_map = {'cube': 0, 'sphere': 1, 'cylinder': 2}
    shape = shape_map.get(object_data['shape'], 0)  # default to cube if shape is not found

    size = object_data['size']
    position = object_data['position']
    color = object_data.get('color_id', [255, 255, 255])  # default color is white if not specified
    color = [i / 255 for i in color]

    # Create shape
    handle = sim.createPureShape(shape, 16, size, 0.1)

    # Check if the object was created successfully
    if handle != -1:
        sim.setObjectPosition(handle, -1, position)
        sim.setShapeColor(handle, None, sim.colorcomponent_ambient_diffuse, color)  # Set color of object
        return handle
    else:
        print(f"Failed to create {object_data['shape']} with id {object_data['id']}.")
        return None

# Function to create and add a vision sensor
def create_vision_sensor():
    # Define vision sensor parameters
    options = 2  # Specify options for the vision sensor
    intParams = [512, 512, 0, 0]  # Increased resolution for better image quality
    floatParams = [
        0.01,  # Near clipping plane
        2.0,  # Increased far clipping plane
        1.0,  # Wider view angle
        0.1,  # Sensor size x
        0.0,  # Reserved
        0.0,  # Reserved
        1.0,  # Null pixel red-value
        1.0,  # Null pixel green-value
        1.0,  # Null pixel green-value
        0.0,  # Reserved
        0.0  # Reserved
    ]
    # Attempt to create the vision sensor
    sensor_handle = sim.createVisionSensor(options, intParams, floatParams)
    # Check if the sensor was created successfully
    if sensor_handle != -1:
        with open("config/config.json") as f:
            config = json.load(f)
            head_position = config["config"]["head_position"]
        f.close()
        # Position the sensor above the scene
        position = head_position
        sim.setObjectPosition(sensor_handle, -1, position)

        # Set the orientation of the sensor (optional)
        orientation = [0, 2.5 * np.pi / 2, -np.pi / 2]  # Roll, Pitch, Yaw
        sim.setObjectOrientation(sensor_handle, -1, orientation)

        return sensor_handle
    else:
        print("Failed to create vision sensor.")
        return None

# Function to reset the simulation for each new scene
def reset_simulation():
    sim.stopSimulation()
    time.sleep(1)  # Wait to ensure simulation has fully stopped
    sim.startSimulation()
    time.sleep(1)  # Wait for simulation to initialize

    # Create a new vision sensor each time the simulation is reset
    return create_vision_sensor()

# Function to capture and save an image from the vision sensor
def capture_scene_image(scene_id, sensor_handle, folder_path):
    img, resX, resY = sim.getVisionSensorCharImage(sensor_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    if img is not None:
        image = np.flip(img, axis=0)  # Flip vertically to correct the image orientation
        img = Image.fromarray(image, 'RGB')
        img.save(os.path.join(folder_path, f"scene_{scene_id}.png"))
        print(f"Image for scene {scene_id} saved as scene_{scene_id}.png.")
    else:
        print(f"Failed to capture image for scene {scene_id}.")

# Function to create and save the scene and image
def create_and_save_scene(scene, folder_path_scenes, folder_path_images):
    scene_id = scene['scene_id']
    print(f"Creating scene: {scene_id}")

    # Reset the simulation to ensure a clean start and create a vision sensor
    vision_sensor_handle = reset_simulation()

    object_handles = []
    for obj in scene['objects']:
        handle = create_object(obj)
        if handle:
            object_handles.append(handle)

    # Save the scene in CoppeliaSim
    sim.saveScene(os.path.join(folder_path_scenes, f"scene_{scene_id}.ttt"))  # Save the scene in CoppeliaSim format
    time.sleep(0.5)  # Wait a moment to ensure the scene is saved

    # Capture and save an image of the scene
    capture_scene_image(scene_id, vision_sensor_handle, folder_path_images)
    print(f"Scene {scene_id} created with {len(object_handles)} objects and saved as scene_{scene_id}.ttt.")

# Main function to orchestrate the scene creation process
def main(path2data):
    # Load the scene from a JSON file
    scene_data = load_scene_from_json(path2data+'/scenes_with_positions.json')

    folder_path_scenes = os.path.join(path2data, f"scenes_ttt")
    folder_path_images = os.path.join(path2data, f"scenes_imgs")

    # Create folders if they do not exist
    os.makedirs(folder_path_scenes, exist_ok=True)
    os.makedirs(folder_path_images, exist_ok=True)

    # Iterate over each scene and create objects
    for scene in scene_data['Scenes']:
        create_and_save_scene(scene, folder_path_scenes, folder_path_images)

if __name__ == "__main__":
    sizes = ["small", "medium", "large"]
    for size in sizes:
        for i in range(4):
            main(f"data/{size}_eval/scenes_type_{i}")
            main(f"data/{size}_train/scenes_type_{i}")
            main(f"data/{size}_test/scenes_type_{i}")
