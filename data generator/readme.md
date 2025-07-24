# Dataset Creation

This project is designed to create a dataset based on specified scenes. The dataset generation process is handled by the `create_data.py` script.

## Project Structure

- `create_data.py` — the main script for data generation
- `config/` — folder containing configuration files for each scene
- `data/` — folder where the generated data is saved

## How to Create the Dataset

1. **Add a Configuration File**  
   Create a configuration file for the specific scene inside the `config/` folder. This file defines parameters for data generation (e.g., scene type, objects, actions, maximum radius, etc.).

2. **Run the Script**  
   Execute the following command in the terminal:

   ```bash
   python create_data.py 
