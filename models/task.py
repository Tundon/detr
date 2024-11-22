import torch
import lightning as L

from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from . import detr


class DETRDetection(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 91,
        num_queries: int = 100,
        nheads: int = 8,
        hidden_dim: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        # Image size
        height: int = 800,
        width: int = 1200,
        log_losses: Optional[List[str]] = [
            "loss",
            "loss_bbox",
            "loss_giou",
            "loss_ce",
            "class_error",
            "cardinality_error",
        ],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # CLI args from official DETR
        args = SimpleNamespace(
            device="cuda",
            lr=3e-05,
            lr_backbone=3e-06,
            batch_size=2,
            weight_decay=0.0001,
            epochs=150,
            lr_drop=100,
            clip_max_norm=0.1,
            frozen_weights=None,
            backbone="resnet50",
            dilation=False,
            position_embedding="sine",
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=2048,
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            num_queries=100,
            pre_norm=False,
            masks=False,
            aux_loss=True,
            set_cost_class=1,
            set_cost_bbox=5,
            set_cost_giou=2,
            mask_loss_coef=1,
            dice_loss_coef=1,
            bbox_loss_coef=5,
            giou_loss_coef=2,
            eos_coef=0.1,
            dataset_file="coco",
            coco_path="data/coco",
        )

        self.model, self.criterion, *_ = detr.build(args)

        self.metrics = MeanAveragePrecision(
            iou_type="bbox",
            box_format="xywh",
            # `faster_coco_eval` will be ~10s faster for the coco val set.
            backend="faster_coco_eval",
        )

    def forward(self, x):
        outputs = self.model(x)

    def training_step(self, batch, batch_index) -> Mapping[str, Any]:
        x, y = batch
        batch_size = len(x)

        outputs = self.model(x)
        losses = self.criterion(outputs, y)

        self._log_losses(losses, batch_size)

        return losses

    def validation_step(self, batch):
        x, y = batch
        batch_size = len(x)

        preds = self.model(x)
        losses = self.criterion(preds, y)
        preds, targets = prepare_torch_metrics(preds, y)
        self.metrics.update(preds, targets)

        # Log validation losses
        for k, v in losses.items():
            if k in self.hparams.log_losses:
                self.log(
                    f"val_{k}", v, sync_dist=True, on_epoch=True, batch_size=batch_size
                )

    def on_validation_epoch_end(self) -> None:
        result = self.metrics.compute()
        self.metrics.reset()
        mAP, mAP_s, mAP_m, mAP_l = (
            result["map"],
            result["map_small"],
            result["map_medium"],
            result["map_large"],
        )
        print(f"[mAP/(s,m,l)]: {mAP:.3f}/({mAP_s:.3f},{mAP_m:.3f},{mAP_l:.3f})")
        self.log_dict(
            {
                "mAP": mAP.to(self.device),
                "mAP_s": mAP_s.to(self.device),
                "mAP_m": mAP_m.to(self.device),
                "mAP_l": mAP_l.to(self.device),
            },
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        # Get all parameters from the model
        all_params = list(self.model.parameters())

        # Get parameters from the backbone
        backbone_params = list(self.model.backbone.parameters())

        # Use named_parameters to ensure correct exclusion
        backbone_param_ids = {id(p) for p in backbone_params if p.requires_grad}
        other_params = [
            p for p in all_params if id(p) not in backbone_param_ids and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": 3e-6,
                    "name": "backbone",
                },
                {
                    "params": other_params,
                    "name": "model",
                },
            ],
            lr=3e-5,
            weight_decay=1e-4,
        )

        return optimizer

    def _log_losses(self, losses: Dict, batch_size):
        # Log only specific losses
        log_losses = {
            k: v
            for k, v in losses.items()
            if k in self.hparams.log_losses
            # f"{k}_{mode}": v for k, v in losses.items() if k in self.hparams.log_losses
        }
        self.log_dict(log_losses, batch_size=batch_size)


def prepare_torch_metrics(outputs: dict, targets: list):
    """
    Prepare the predictions expected by TorchMetrics.

    This function processes the model outputs and target annotations to prepare the predictions
    in the format expected by TorchMetrics for evaluation. It converts the bounding box format,
    applies softmax to the logits to obtain probabilities, and scales the bounding boxes back to
    the original image size.

    Args:
        outputs (dict): A dictionary containing the DETR model's output logits and bounding boxes.
            - "pred_logits" (Tensor): The predicted class logits for each bounding box.
            - "pred_boxes" (Tensor): The predicted bounding boxes in cxcywh format.
        targets (list): A list of dictionaries containing the target annotations for each image.
            - Each dictionary should contain:
                - "image_size" (Tensor): The original size of the image, in (w, h) order.

    Returns:
        tuple: A tuple containing:
            - results (list): A list of dictionaries with the processed predictions for each image.
                - Each dictionary contains:
                    - "scores" (Tensor): The confidence scores for each bounding box.
                    - "labels" (Tensor): The predicted class labels for each bounding box.
                    - "boxes" (Tensor): The predicted bounding boxes in xywh format, scaled to the original image size.
            - targets (list): The original target annotations.

    References:
        https://github.com/Lightning-AI/torchmetrics/blob/release/stable/examples/detection_map.py
    """

    from torch.nn.functional import softmax
    from torchvision.tv_tensors import BoundingBoxFormat
    from torchvision.transforms.v2.functional import convert_bounding_box_format

    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

    prob = softmax(logits, -1)
    # The last element represents the background class, exclude it
    scores, labels = prob[..., :-1].max(-1)
    # COCO dataset has xywh format
    boxes = convert_bounding_box_format(
        boxes,
        BoundingBoxFormat.CXCYWH,
        BoundingBoxFormat.XYWH,
    )

    # Scale boxes back to original image size
    results = []
    for s, l, b, t in zip(scores, labels, boxes, targets):
        image_size = t["image_size"]
        b = b * image_size.tile(2)
        results.append({"scores": s, "labels": l, "boxes": b})

    return results, targets
