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
run:  
```
python3 train_textBPN.py
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


