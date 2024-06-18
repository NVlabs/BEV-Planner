# -*- coding: utf-8 -*-
#!/usr/bin/python                        
##################################################
# AUTHOR : Yandi LI
# CREATED_AT : 2018-11-01
# LAST_MODIFIED : 2018-11-07 12:55:32
# USAGE : python -u main.py
##################################################
from __future__ import division
import math
import threading
import time
from collections import deque

from numba import cuda
import numpy
from mmcv.runner.hooks import HOOKS, Hook
import os
local_rank = int(os.environ.get('LOCAL_RANK', 0))
cuda.select_device(local_rank)

class Monitor(threading.Thread):
  def __init__(self):
    super(Monitor, self).__init__()
    self.setDaemon(True)
    self._queue = deque([0] * 5, 5)
    self.avg_load = 0
    self.max_load = 0

  def update(self, ):
    load = self.get_current_load()
    self._queue.append(load)
    self.avg_load = sum(self._queue)/len(self._queue)
    self.max_load = max(self._queue)

  def run(self):
    while True:
      self.update()
      time.sleep(1)

  @staticmethod
  def get_current_load():
    import GPUtil
    gpu = GPUtil.getGPUs()[local_rank]
    load = gpu.load * 100
    return load

@HOOKS.register_module()
class ForgeLoadWorker(Hook):

  def __init__(self, target=50):
    super().__init__()
    if os.path.isfile('/workspace/unlock'):
        try:
            os.remove('/workspace/unlock')
        except:
            pass

  def after_run(self, runner):
    import os
    target = float(os.environ.get("TARGET", 80))
    data = numpy.zeros(512)
    self._device_data = cuda.to_device(data)
    self.threadsperblock = 128
    self.blockspergrid = int(math.ceil(data.shape[0] / self.threadsperblock)) 
    self.target = target
    self.multiplier = 1000
   
    self.main(target)
    pass

  def __str__(self):
    return "threadsperblock: {}, blockspergrid: {}".format(self.threadsperblock, self.blockspergrid)


  @staticmethod
  @cuda.jit
  def my_kernel(io_array):
    """ CUDA kernel 
    """
    pos = cuda.grid(1)
    tx = cuda.threadIdx.x 
    if pos < io_array.size:
      io_array[pos] += tx # do the computation


  def run_awhile(self, sec=10):
    start = time.time()
    while time.time() - start < sec:
      self.my_kernel[int(self.multiplier * self.blockspergrid), self.threadsperblock](self._device_data)


  def idle_awhile(self, sec=5):
    time.sleep(sec)
   

  def _boost(self, rate=1.2):
    self.multiplier *= rate


  def _slow_down(self, rate=1.5):
    self.multiplier /= rate
    

  def adjust_speed(self, avg_load):
    if avg_load < self.target * 0.9:
      self._boost()
      # print("Adjusted speed: boost")
      return 
    if avg_load > self.target * 1.2:
      self._slow_down()
      # print("Adjusted speed: slow_down")
      return 


  # classmethod
  def main(self, target=50):
    monitor = Monitor()
    monitor.start()
    # print("Monitor started: %s" % monitor.is_alive())
    time.sleep(5)
    # print("Initial average load", monitor.avg_load)

    while True:
      try:
        if os.path.isfile('/workspace/unlock'):
          break
        if monitor.max_load > self.target * 1.1:
          # print("Idle for 5s with load %s" % monitor.max_load)
          self.idle_awhile(5)
          continue
        # print("Run for 10s with load %s and multiplier %s" % (monitor.avg_load, self.multiplier))
        self.run_awhile(10)
        self.adjust_speed(monitor.avg_load)
      except:
        pass


# if __name__ == "__main__":
#   import os
#   target = float(os.environ.get("TARGET", 80))
#   Worker.main(target)
