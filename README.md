# Prepare the data
## Install required packages
1. Install tensorflow. Refer to tensorflow official site for instructions.
2. Install numpy and PIL by `pip install numpy pillow`
## Download data
The dataset we'll be using is ILSVRC2015 VID. Find the download link on
[ILSVRC](http://image-net.org/challenges/LSVRC/2015/#vid).

Extract the 84G archive to a directory. We assume `$HOME/data/ILSVRC2015` is used.
## Preprocess the data
```sh
python preprocess.py\
           --image_dir="$HOME/data/ILSVRC2015/Data/VID/train"\
           --annotation_dir="$HOME/data/ILSVRC2015/Annotations/VID/train"\
           --output_dir="$HOME/data/ILSVRC2015/cropped/your_output_dir"
```
This will create a series of subdirectories in `$HOME/data/ILSVRC2015/cropped/your_output_dir`, each corresponding to an object in a video.

In each subdirectory, there will be a series of .txt and .bin files. The .txt
file contains metadata about the bounding boxes, the crop boxes and the path to
the actual frame image file. The .bin files are dumped float32 array of heatmaps.

# Running the training
## Getting the frozen vgg weights
1. Get vgg\_16.tar.gz from [tensorflow](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).
2. Untar it to get vgg\_16.ckpt. Put it into current directory.
3. Run `python freeze_vgg.py` to get vgg\_16.pbtxt
4. Use [this](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)
   tool to convert vgg\_16.pbtxt into vgg\_16\_frozen.pb
5. Put vgg\_16\_frozen.pb into current directory.
## Run the training and validation
```sh
python \
direct_offset_output.py \
--batch_size=1 \
--sequence_length=20 \
--num_layers=1 \
--log_interval=1 \
--num_epochs=10 \
--learning_rate=1e-4 \
--hidden_size=200 \
--read_head_size=4 \
--write_head_size=1 \
--mem_dim=20 \
--mem_size=128 \
--sequences_dir=$HOME/data/ILSVRC2015/cropped/your_output_dir \
--tag=<meaningful_suffix_to_the_log_dir> \
--gt_width=8 \
```
Check the specific script for more options.

Namely there are two versions of networks. `direct_offset_output.py` is
implemented with NTM as core. `direct_offset_output_with_dnc.py` is implemented
with DNC as core.

Once the training starts, you can type `sh visualize.sh` to start the
tensorboard.
You may also find the logs under `./logs/time_stamp_plus_your_tag`
