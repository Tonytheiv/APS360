# APS360

## Baseline

In order to run the baseline you must run the `preprocess.ipynb` first, runnning all cells should work.

(Currently only crowds_zara02.avi are present in the data folder)

Running `baseline.ipynb` will use cv2 centroid tracking and frame differencing to identify moving elements of the video. The program will output a processed video in `processsed_videos`

### Info

The `maxDissapeared` arguement in `CentroidTracker()` will change how long objects persist in "memory," line 84 in cell 1 in `baseline.ipynb` sets this to 0, the default is 50.