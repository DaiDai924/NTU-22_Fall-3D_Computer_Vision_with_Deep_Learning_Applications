# Modified MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation

## Installation

- Create a conda environment: ```conda create -n mhformer python=3.7```
- Download cudatoolkit=11.0 from [here](https://developer.nvidia.com/cuda-11.0-download-archive) and install 
- ```pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html```
- ```pip3 install -r requirements.txt```


## Test our best model

To test on a 27-frames pretrained model on Human3.6M:

```bash
python main.py --test --previous_dir 'checkpoint/ours/best_model' --frames 27
```

Here, we compare our best model and the original MHFormer on Human3.6M dataset. Evaluation metric is Mean Per Joint Position Error (MPJPE) in mm​. 


|   Models    |  MPJPE (mm) |
| :---------: | :---------: |
|   MHFormer  |    46.88    |
|   Our Best  |  **46.33**  |


## Train our best model

To train a 27-frames model on Human3.6M:

```bash
python main.py --frames 27 --nepoch 10 --batch_size 256 --loss_fn [loss_fn]
```


## Plot result

To plot the results of our model and the original MHFormer. 

```bash
python plot_result.py --result_dir 'checkpoint'
```


## Citation

If you find our work useful in your research, please consider citing:

    @inproceedings{li2022mhformer,
      title={MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation},
      author={Li, Wenhao and Liu, Hong and Tang, Hao and Wang, Pichao and Van Gool, Luc},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={13147-13156},
      year={2022}
    }
  