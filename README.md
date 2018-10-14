# mAIson
Deep learning to identify game metadata in Call of Duty: Black Ops 4

Template matching is poor, and pretty boring. There is too much gray that confuses the pattern matcher. 

In multiplayer, the background behind each gun is just barely transparent, causing it to be only slightly modified by the background in the game. By collecting hundreds of images of each gun across different maps, we should be able to train a DNN to identify the gun being used.

This learning can be extended to identify other key elements that are displayed in different game modes, win/loss scenarios, etc. By getting tons of images of those items and labeling as such, maybe the DNN will be able to generally identify those components. We can help the model by cropping the images of the game to those specific parts of the screen so that instead of a 1440p/1080p image, the model only works against a much smaller region. This is also faster!

## Data Collection
Get in-game screenshots of every weapon across different maps/gamemodes
Find bounding-box locations where a weapon image is located
Store into numpy serialized files
Run batched model fit