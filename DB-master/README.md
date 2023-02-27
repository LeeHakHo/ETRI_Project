# Introduction
This is a PyToch implementation of DBNet([arxiv](https://arxiv.org/abs/1911.08947)) and DBNet++([TPAMI](https://ieeexplore.ieee.org/abstract/document/9726868/), [arxiv](https://arxiv.org/abs/2202.10304)).  It presents a real-time arbitrary-shape scene text detector, achieving the state-of-the-art performance on standard benchmarks.

Part of the code is inherited from [MegReader](https://github.com/Megvii-CSG/MegReader).

## Installation

### Requirements:
- Python3.9
- PyTorch == 1.2 
- GCC >= 4.9 (This is important for PyTorch)
- CUDA >= 9.0 (10.1 is recommended)

# python dependencies
pip install -r requirement.txt

# install PyTorch with cuda-10.1
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

```

## Models
New: DBNet++ trained models [Google Drive](https://drive.google.com/drive/folders/1buwe_b6ysoZFCJgHMHIr-yHd-hEivQRK?usp=sharing).

Download Trained models [Baidu Drive](https://pan.baidu.com/s/1vxcdpOswTK6MxJyPIJlBkA) (download code: p6u3), [Google Drive](https://drive.google.com/open?id=1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG).

### Config file
**The YAML files with the name of ```base*.yaml``` should not be used as the training or testing config file directly.**

### Demo
Run the model inference with a single image. Here is an example:
```

```CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --image_path datasets/total_text/test_images/img10.jpg --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7 --visualize```

The results can be find in `demo_results`.

```box_thresh``` can be used to balance the precision and recall, which may be different for different datasets to get a good F-measure. ```polygon``` is only used for arbitrary-shape text dataset. The size of the input images are defined in ```validate_data->processes->AugmentDetectionData``` in ```base_*.yaml```.


```CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet18_deform_thre.yaml --resume path-to-model-directory/totaltext_resnet18 --polygon --box_thresh 0.7 --speed```

Note that the speed is related to both to the GPU and the CPU since the model runs with the GPU and the post-processing algorithm runs with the CPU.

## Training
Check the paths of data_dir and data_list in the base_*.yaml file. For better performance, you can first per-train the model with SynthText and then fine-tune it with the specific real-world dataset.

```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4```

You can also try distributed training (**Note that the distributed mode is not fully tested. I am not sure whether it can achieves the same performance as non-distributed training.**)

```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py path-to-yaml-file --num_gpus 4```

## Citing the related works

Please cite the related works in your publications if it helps your research:

     @inproceedings{liao2020real,
      author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
      title={Real-time Scene Text Detection with Differentiable Binarization},
      booktitle={Proc. AAAI},
      year={2020}
    }

    @article{liao2022real,
      title={Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion},
      author={Liao, Minghui and Zou, Zhisheng and Wan, Zhaoyi and Yao, Cong and Bai, Xiang},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2022},
      publisher={IEEE}
    }


    

