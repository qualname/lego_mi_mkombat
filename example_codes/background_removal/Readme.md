## Requirements

OpenCV is required to run this example! (`$ python3 -m pip install opencv`)  
If you don't want to install OpenCV, replacing it with another library that can open `.png` files (e.g. `Pillow`) should work too. `cv2.cvtColor` and `cv2.threshold` may be replaced with `numpy` slices.

## Output

The code creates a grayscale (binary) mask (`mask.gif`). Merging the mask and the observation will yield `masked_observation.gif`.  
Note that some moving elements of the background (the water lillies in this case) may remain. Removing those, while possible (check `final_output.gif`), is computationally intense, and will drop some of the character outline details too, so it is not included in this code.
