from mmcv.utils import Registry, is_method_overridden
from mmcv.runner.hooks import HOOKS, CheckpointHook, Hook
from mmcv.runner.dist_utils import allreduce_params, master_only
import time

@HOOKS.register_module()
class TimerCP(CheckpointHook):

    # designed for NVIDIA ORD, each job can only run for 4 hours.
    # period = 4h = 4 * 3600
    def __init__(self, period=14400):
        super().__init__()
        self.period = period - 180 # 3 mins redundancy
        self.not_save = True

    def before_run(self, runner):
        super().before_run(runner)
        self.start_time = time.time()

    def after_train_epoch(self, runner):
        pass

    def before_train_iter(self, runner):
        running_time = (time.time() - self.start_time)
        if running_time > self.period and self.not_save:
            runner.logger.info(
                f'TimerCP: Saving checkpoint at {runner.iter + 1} iterations. Period: '+'%.1fh' % (self.period/3600)
                )
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)
            self.not_save = False

    @master_only
    def _save_checkpoint(self, runner):
        super()._save_checkpoint(runner)
  


 