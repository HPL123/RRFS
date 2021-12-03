from ..registry import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.core import auto_fp16, get_classes, tensor2imgs
import numpy as np

@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.bbox_head.gzsd = test_cfg.rcnn.gzsd
    
    def classfier(self, x):

        # shared part os zero for faster rcnn
        # if self.bbox_head.num_shared_convs > 0:
        #     for conv in self.shared_convs:
        #         x = conv(x)
        
        # separate branches
        x_cls = x

        for conv in self.bbox_head.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.bbox_head.with_avg_pool:
                x_cls = self.bbox_head.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.bbox_head.cls_fcs:
            x_cls = self.bbox_head.relu(fc(x_cls))

        cls_score = self.bbox_head.fc_cls(x_cls) if self.bbox_head.with_cls else None
        return cls_score


        # cls_score = self.bbox_head.fc_cls(x)
        # return cls_score
    
    @auto_fp16(apply_to=('img', ))
    def forward(self, img=None, img_meta=None, return_loss=True, feats=None, classifier_only=False, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=False`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=True`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if classifier_only:
            return self.classfier(feats)
        elif return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            

            cls_score, bbox_pred, feats_cls = self.bbox_head(bbox_feats, return_cls_feats=True)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets, feats_cls=feats_cls)
            
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    


    def feats_extract(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        img = img.cuda()
        x = self.extract_feat(img)


        # RPN forward and loss
        if self.with_rpn:

            rpn_outs = self.rpn_head(x)
            # rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                        #   self.train_cfg.rpn)
            # rpn_losses = self.rpn_head.loss(
            #     *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            # losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)

            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i].cuda(),
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i].cuda())
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i].cuda(),
                    gt_labels[i].cuda(),
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss

        # if self.with_bbox:
        rois = bbox2roi([res.bboxes for res in sampling_results])
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        
        # bbox_feats-->shape ==> num_boxes x 5
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        # cls_score, bbox_pred = self.bbox_head(bbox_feats)
        # aaa = self.classfier(bbox_feats)

        if self.bbox_head.num_shared_fcs > 0:
            # already avg_pooled 
            # if self.with_avg_pool:
            #     x = self.avg_pool(x)
            bbox_feats = bbox_feats.view(bbox_feats.size(0), -1)
            for fc in self.bbox_head.shared_fcs:
                bbox_feats = self.bbox_head.relu(fc(bbox_feats))

        bbox_targets = self.bbox_head.get_target(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)

        bg_inds = np.where(bbox_targets[0].data.cpu().numpy()==0)[0]
        fg_inds = np.where(bbox_targets[0].data.cpu().numpy()>0)[0]
        #bg_scores = cls_score[:, 0]
        #sorted_args = np.argsort(bg_scores.data.cpu().numpy(), kind='mergesort')[:len(fg_inds)*3]
        #selected_bg_inds = np.intersect1d(sorted_args, bg_inds)
        sub_neg_inds = np.random.permutation(bg_inds)[:int(2*len(fg_inds))]
        # 
        inds_to_select = np.concatenate((sub_neg_inds, fg_inds))
        return bbox_feats[inds_to_select], bbox_targets[0][inds_to_select], bbox_targets[2][inds_to_select]
        # return bbox_feats, bbox_targets[0], bbox_targets[2]
