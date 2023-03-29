#!/bin/bash

rm -rf "../work_dirs"
rm -rf "../visual"

# check if a directory name was passed as an argument
if [ $# -eq 0 ]; then
  echo "Error: Please provide the name of the directory to create a symbolic link for."
  exit 1
fi

# assign the first argument to a variable
dir_name=$1

# Define the folder you want to check and create
folder_name="your_folder"

# Check if the folder exists
if [ -d "/home/ubuntu/storage/weightDir/experiment/$dir_name" ]; then
    echo "Folder '$dir_name in weightDir' already exists."
else
    # create a work_dir directory under the specified path
    mkdir "/home/ubuntu/storage/weightDir/experiment/$dir_name"
    echo "Folder '$dir_name in weightDir' created."
fi

# mkdir "/home/ubuntu/storage/weightDir/experiment/$dir_name"

# create a symbolic link to the directory
ln -s "/home/ubuntu/storage/weightDir/experiment/$dir_name" "work_dirs"



# Check if the folder exists
if [ -d "/home/ubuntu/storage/visual/experiment/$dir_name" ]; then
    echo "Folder '$dir_name in visual' already exists."
else
    # create a visual directory under the specified path
    mkdir "/home/ubuntu/storage/visual/experiment/$dir_name"
    echo "Folder '$dir_name in visual' created."
fi

# # create a visual directory under the specified path
# mkdir "/home/ubuntu/storage/visual/experiment/$dir_name"

# create a symbolic link to the directory
ln -s "/home/ubuntu/storage/visual/experiment/$dir_name" "visual"

# create a folder for config
mkdir -p "configs/swp"  
echo " swp config dir in $dir_name  has been created"

# create a folder for learn
mkdir -p "learn"


ln -s "../splCode/tempDataset" "tempDataset"

echo "learn dir in $dir_name  has been created"
