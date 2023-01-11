import torch
import os
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(os.getcwd())

#print(paddle.utils.run_check()