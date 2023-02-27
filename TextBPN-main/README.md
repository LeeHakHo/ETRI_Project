# TextBPN
Adaptive Boundary Proposal Network for Arbitrary Shape Text Detection； Accepted by ICCV2021.  
![](https://github.com/GXYM/TextBPN/blob/main/vis/1.png)  

## 1.Prerequisites 
  python 3.9;  
  PyTorch 1.7.0;   
  Numpy >=1.2.0   
  CUDA 11.1;  
  GCC >=10.0;  
  *opencv-python < 4.5.0*  
  NVIDIA GPU(with 11G or larger GPU memory for inference); 
  
## 2.Arguments
```
You have to go to configs.CDistNet_config.py and edit it yourself.

dist_vocab: Text path of character to use as class
dist_vocab_size: The number of characters to use as a class
train.gt_file: path to train lmdb
train.num_epochs: num of train epochs
val.gt_file: path to evaulation lmdb
test.best_acc_test: Whether to use the best accuarcy pth file - mine
test.s_epoch,e_epoch, avg_e: epochs of the model.pth file to be tested
test.test_list: ldmb path to test**
test.model_dir: model.pth path to test
test.save_dir: Path to save test results
test.python_path: Your python path
```  

## 3.Train
run:  
```
python3 train_textBPN.py
```

## 4.Demo
run:  
```
python3 demo.py
```


## Citing the related works

Please cite the related works in your publications if it helps your research:
``` 
  @inproceedings{DBLP:conf/iccv/Zhang0YWY21,
  author    = {Shi{-}Xue Zhang and
               Xiaobin Zhu and
               Chun Yang and
               Hongfa Wang and
               Xu{-}Cheng Yin},
  title     = {Adaptive Boundary Proposal Network for Arbitrary Shape Text Detection},
  booktitle = {2021 {IEEE/CVF} International Conference on Computer Vision, {ICCV} 2021, Montreal, QC, Canada, October 10-17, 2021},
  pages     = {1285--1294},
  publisher = {{IEEE}},
  year      = {2021},
}

@inproceedings{Zhang2022ArbitraryST,
  title={Arbitrary Shape Text Detection via Boundary Transformer},
  author={S. Zhang and Xiaobin Zhu and Chun Yang and Xu-Cheng Yin},
  year={2022}
}
  ``` 
 
 ## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/GXYM/DRRG/blob/master/LICENSE.md) file for details


