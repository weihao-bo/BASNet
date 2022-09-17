"""
ModelName: test
Description: 
Author：bwh
Date：2022/2/5 15:40
"""
import os
import time
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import dataset as dataset
from net.BASNet import BASNet


#torch.cuda.set_device(1)
class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg = Dataset.Config(datapath=path, snapshot='../out/BASNet/CHINA/model', mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            cost_time = list()
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                out = self.net(image, shape)

                torch.cuda.synchronize()
                cost_time.append(time.perf_counter() - start_time)
                pred = np.squeeze((torch.sigmoid(out[0]) * 255).cpu().numpy())

                save_path = '../res/BASNet/CHINA'

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                cv2.imwrite(save_path + '/' + name[0] + '.png', np.round(pred))

            cost_time.pop(0)
            print('Mean running time is: ', np.mean(cost_time))
            print("FPS is: ", len(self.loader.dataset) / np.sum(cost_time))


if __name__ == '__main__':
    for path in ['../data/CHINA-FIRE']:
        test = Test(dataset, BASNet, path)
        test.save()