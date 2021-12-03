from mmcv.runner import Runner
import logging
from mmcv.runner.utils import get_host_info, get_time_str, obj_from_dict
import mmcv
import time
import torch
from collections import OrderedDict
from splits import get_unseen_class_ids, get_seen_class_ids
from mmdet.apis import get_root_logger

def copy_synthesised_weights(model, filename, dataset_name='voc', split='65_15'):

    logger = get_root_logger('INFO')

    checkpoint = torch.load(filename, map_location='cpu')
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))
    
    if hasattr(model, 'module'):
        own_state = model.module.state_dict()
    else:
        own_state = model.state_dict()

    unseen_class_inds = get_unseen_class_ids(dataset=dataset_name, split=split)
    seen_bg_weights = own_state['bbox_head.fc_cls.weight'][0].data.cpu().numpy().copy()
    seen_bg_bias = own_state['bbox_head.fc_cls.bias'][0].data.cpu().numpy().copy()

    ##TODO visualize
    own_state['bbox_head.fc_cls.bias'][unseen_class_inds] = state_dict['fc1.bias'][1:]
    own_state['bbox_head.fc_cls.weight'][unseen_class_inds] = state_dict['fc1.weight'][1:]
    ##visualize
    # own_state['bbox_head.fc_cls.bias'][unseen_class_inds] = state_dict['fc1.bias'][1:].cuda()
    # own_state['bbox_head.fc_cls.weight'][unseen_class_inds] = state_dict['fc1.weight'][1:].cuda()
    ##

    alpha1 = 0.35
    alpha2 = 0.65
    own_state['bbox_head.fc_cls.bias'][0] = alpha1*own_state['bbox_head.fc_cls.bias'][0] + alpha2*state_dict['fc1.bias'][0]
    ##todo visualize
    own_state['bbox_head.fc_cls.weight'][0] = alpha1 * own_state['bbox_head.fc_cls.weight'][0] + alpha2 * \
                                              state_dict['fc1.weight'][0]
    ##visualize
    # own_state['bbox_head.fc_cls.weight'][0] = alpha1 * own_state['bbox_head.fc_cls.weight'][0] + alpha2 * \
    #                                           state_dict['fc1.weight'][0].cuda()
    ##
    
    logger.info(f'{dataset_name} {unseen_class_inds.shape} seenbg: {alpha1} synbg: {alpha2} copied classifier weights from {filename} \n')
    return seen_bg_weights, seen_bg_bias
class Runner2(Runner):
    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 eval_only=False):
        super(Runner2, self).__init__(model, batch_processor, optimizer, work_dir, log_level, logger)
        self.eval_only = eval_only

    def get_unseen_class_inds(self, dataset='voc'):
        return get_unseen_class_ids(dataset)
    def copy_syn_weights(self, filename, split='65_15'):
        copy_synthesised_weights(self.model, filename, split=split )

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.
        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')


        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')


    
