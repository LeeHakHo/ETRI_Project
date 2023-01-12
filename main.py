import torch
import os
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(os.getcwd())

torch.cuda.empty_cache()

import gc
gc.collect()

#print(paddle.utils.run_check()