import glob
import rerun as rr

# init sdk and spawn a local Viewer
rr.init("rrd_visualizer", spawn=True)

# load each .rrd file into the Viewer
for rrd_path in glob.glob("/scratch/km6748/vision-experiments/outputs/2025-06-09/10-19-54/valid/*.rrd"):
    rr.log_file_from_path(rrd_path)
