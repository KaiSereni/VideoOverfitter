# Video Overfitter
**WATCH THIS VIDEO FOR CONTEXT**: [https://youtu.be/-9GCtKRqYP8](https://youtu.be/-9GCtKRqYP8)

## About
This script trains a neural network that takes noise as input and creates an "AI-generated video" using a training dataset of one video. I have a quickstart for experienced Python users, and a full description of how to run the script for those without coding experience. I am highly open to merging pull requests that optimize the script, create a UI or more intuitive CLI, add features, etc.

## Quickstart (Experienced Python users)
1. Clone the repo.
2. Create a Python venv in the repo directory and install the requirements.
3. Convert your video to a **1:1 aspect ratio GIF** and place it in the repo's root. Name it `input_video.gif`.
4. Run `train.py` with the venv. If the video looks weird, increase the `NUM_EPOCHS` variable.

## Full tutorial (New Python users)
1. Install [Python](https://python.org)
2. On the repo page, click the green `Code` button, then click download zip.
3. Right click the zip and press `Extract` to save the folder with the code to your computer (this will be the repo folder).
4. Convert your video to a **1:1 aspect ratio GIF** and place it in the repo folder. Name it `input_video.gif`. 

### If you're on Windows:
5. Navigate to the repo folder in File Explorer, click on the path at the top to highlight it, then press ctrl+c to copy the full repo path.
6. Open **PowerShell**, then type `cd ` followed by the path name, pasting it after cd with ctrl+v. Then press enter to run the command.
7. Type `python -m venv venv` followed by `venv/Scripts/activate`.
8. Run `python train.py`. This will take quite some time depending on the size of the video file. 

### If you're on Mac:
5. Navigate to the repo folder and right click (two finger click on trackpad) the `train.py` file. Press `get info`. Under `General > Where:`, you'll see the folder's path. Right click on it and press `copy as pathname`.
6. Open the `Terminal` app and type `cd ` followed by the past name, pasting it after cd with cmd+v. Then press enter to run the command.
7. Run `python -m venv venv` followed by `source venv/bin/activate`. 
8. Run `python train.py`. This will take quite some time depending on the size of the video file. 
