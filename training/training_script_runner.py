#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
from pathlib import Path

# Change working directory to the location of this script
script_directory = Path(__file__).resolve().parent
os.chdir(script_directory)

# Setting home dirrectory for the project
home_dir = Path.cwd().parent

# Directory where configuration files are stored
config_dir = f"{home_dir}/training/configs"


# Get a list of all .ini files in the directory
config_files = [os.path.join(config_dir, file) for file in os.listdir(config_dir) if file.endswith('.ini')]
print(config_files)

# # List of .ini configuration file paths
# config_files = [
#     'config1.ini',
#     'config2.ini',
#     'config3.ini',
#     'config4.ini',
#     'config5.ini'
# ]

# Path to the script to run
script_path = 'training_script.py'

# Loop through each configuration file
for config_file in config_files:
    print(f"Running {script_path} with configuration {config_file}")
    
    # Run the script with the current configuration
    subprocess.run(['python', script_path, '--config', config_file], check=True)
    
    print(f"Finished running with configuration {config_file}\n")