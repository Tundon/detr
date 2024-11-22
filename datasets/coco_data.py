import torch
import torchvision
import torchvision.transforms.v2 as T

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from torchvision import datasets
from torchvision.transforms.v2 import functional as F
from collections import defaultdict
from torchvision import tv_tensors
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from typing import Any, Dict
import torch
import lightning as L

from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import tv_tensors
from torchvision.transforms import v2


def collate(batch):
    """
    The dataloader collate function.

    This collate function would batch the images into a tensor, while making
    the targets a list
    """
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


class CocoLikeDetection(datasets.CocoDetection):
    """
    A dataset class for COCO-like object detection that extends the standard COCO dataset.

    This dataset allows for custom handling of target keys and their processing using handlers.

    Args:
        root (str | Path): Root directory where images are downloaded to.
        annFile (str): Path to json annotation file.
        transform (Callable[..., Any] | None, optional): A function/transform that takes in a PIL image and returns a transformed version. Default is None.
        target_transform (Callable[..., Any] | None, optional): A function/transform that takes in the target and transforms it. Default is None.
        transforms (Callable[..., Any] | None, optional): A function/transform that takes input sample and its target as entry and returns a transformed version. Default is None.
        target_keys (Tuple[str], optional): A tuple of target keys to be processed. Default is ("boxes", "labels").
        target_handlers (Optional[Dict[str, Callable]], optional): A dictionary of handlers for processing target keys. Default is None.

    Attributes:
        target_keys (Tuple[str]): The keys in the target to be processed.
        target_handlers (Dict[str, Callable]): Handlers for processing target keys. Default handlers include:
            - "image_id": handle_image_id
            - "image_size": handle_image_size
            - "boxes": handle_boxes
            - "labels": handle_labels

    Methods:
        _to_batched_target(target: List[Dict]) -> Dict[str, List]:
            Converts a list of dictionaries to a dictionary of lists.

        __getitem__(index: int) -> Tuple[Any, Any]:
            Retrieves an image and its corresponding target at the specified index.

        get_label_name(label: int) -> str:
            Retrieves the name of a label given its ID.
    """

    def __init__(
        self,
        root: str | Path,
        annFile: str,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        transforms: Callable[..., Any] | None = None,
        target_keys: Tuple[str] = ("boxes", "labels"),
        target_handlers: Optional[Dict[str, Callable]] = None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)

        self.target_keys = target_keys

        default_handlers = {
            "image_id": handle_image_id,
            "image_size": handle_image_size,
            "boxes": handle_boxes,
            "labels": handle_labels,
        }
        # Merge default handlers with user-provided handlers
        self.target_handlers = {**default_handlers, **(target_handlers or {})}

    def _to_batched_target(self, target: List[Dict]):
        """
        Converts a list of dictionaries to a dictionary of lists.

        Args:
            target (List[Dict]): A list of dictionaries containing target annotations.

        Returns:
            Dict[str, List]: A dictionary where each key contains a list of values from the input dictionaries.
        """

        dict_of_lists = defaultdict(list)
        for dct in target:
            for key, value in dct.items():
                dict_of_lists[key].append(value)

        # Assert that all lists in dict_of_lists have the same length
        lengths = [len(v) for v in dict_of_lists.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                "All keys in the target must have the same number of elements"
            )
        return dict(dict_of_lists)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        image_id = self.ids[index]
        image = self._load_image(image_id)

        # `target` is a list of dict including
        # the fields in the object detection annotation in here:
        # https://cocodataset.org/#format-data
        target = self._load_target(image_id)

        # And process the target here
        batched_target = self._to_batched_target(target)
        original_target = target
        target = {}
        # Process additional keys using handlers
        for key in self.target_keys:
            if key in self.target_handlers:
                handler = self.target_handlers[key]
                handler(
                    self,
                    index=index,
                    key=key,
                    image=image,
                    target=target,
                    batched_target=batched_target,
                    original_target=original_target,
                )

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_label_name(self, label: int) -> str:
        """
        Retrieves the name of a label given its ID.

        Args:
            label (int): The ID of the label.

        Returns:
            str: The name of the label.
        """
        return self.coco.cats[label]["name"]


def handle_image_id(
    dataset: datasets.CocoDetection, index: int, key: str, target, **kwargs
):
    """
    Handler for processing the image ID.

    Args:
        dataset (datasets.CocoDetection): The dataset instance.
        index (int): The index of the image.
        key (str): The key to store the image ID in the target.
        target (dict): The target dictionary to update.
    """
    image_id = dataset.ids[index]
    target[key] = image_id


def handle_image_size(
    dataset: datasets.CocoDetection,
    index: int,
    key: str,
    target,
    image: Image.Image,
    **kwargs,
):
    """Put the image size into target["image_size"]."""
    target["image_size"] = torch.tensor(image.size)


def handle_boxes(
    dataset: datasets.CocoDetection,
    index: int,
    key: str,
    image,
    target,
    batched_target,
    **kwargs,
):
    """
    Handler for processing bounding boxes.

    Args:
        dataset (datasets.CocoDetection): The dataset instance.
        index (int): The index of the image.
        key (str): The key to store the bounding boxes in the target.
        image (Any): The image corresponding to the target.
        target (dict): The target dictionary to update.
        batched_target (dict): The batched target annotations.
    """
    canvas_size = tuple(F.get_size(image))
    if "bbox" in batched_target:
        target[key] = tv_tensors.BoundingBoxes(
            batched_target["bbox"],
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=canvas_size,
        )
    else:
        target[key] = tv_tensors.BoundingBoxes(
            torch.empty((0, 4)),
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=canvas_size,
        )


def handle_labels(
    dataset: datasets.CocoDetection,
    index: int,
    key: str,
    target,
    batched_target,
    original_target,
    **kwargs,
):
    """
    Handler for processing labels.

    Args:
        dataset (datasets.CocoDetection): The dataset instance.
        index (int): The index of the image.
        key (str): The key to store the labels in the target.
        target (dict): The target dictionary to update.
        batched_target (dict): The batched target annotations.
        original_target (list): The original target annotations.
    """
    labels = [item["category_id"] for item in original_target]
    target[key] = torch.tensor(labels, dtype=torch.int64)


class ConvertBox(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, out_fmt="", normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            "xyxy": BoundingBoxFormat.XYXY,
            "cxcywh": BoundingBoxFormat.CXCYWH,
        }

    def _transform(self, inpt: BoundingBoxes, params: Dict[str, Any]) -> Any:

        canvas_size = inpt.canvas_size
        if self.out_fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(
                inpt, in_fmt=in_fmt, out_fmt=self.out_fmt
            )
            inpt = BoundingBoxes(
                inpt, format=self.data_fmt[self.out_fmt], canvas_size=canvas_size
            )

        if self.normalize:
            # To normalize the bbox, [w,h] needs to be expanded to [w,h,w,h] so
            # that broadcasting could work with multiple bboxes.
            # Note, the canvas_size is (h, w)
            inpt = inpt / torch.tensor(canvas_size[::-1]).tile(2)
        return inpt


class COCO(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for COCO-like object detection dataset.

    Args:
        batch_size (int): Number of samples per batch.
        train_data_path (str): Path to the training data.
        train_ann_file (str): Path to the training annotations file.
        val_data_path (str): Path to the validation data.
        val_ann_file (str): Path to the validation annotations file.
        image_resize (Tuple[int, int]): Desired image size (h, w) for resizing.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 8.
    """

    def __init__(
        self,
        batch_size: int,
        image_resize: Tuple[int, int],
        train_data_path: str = "data/coco/train2017",
        train_ann_file: str = "data/coco/annotations/instances_train2017.json",
        val_data_path: str = "data/coco/val2017",
        val_ann_file: str = "data/coco/annotations/instances_val2017.json",
        num_workers: int = 8,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = CocoLikeDetection(
            root=train_data_path,
            annFile=train_ann_file,
            transforms=v2.Compose(
                [
                    v2.ToImage(),
                    v2.RandomZoomOut(
                        fill={tv_tensors.Image: (123, 117, 104), "others": 0}
                    ),
                    v2.RandomIoUCrop(),
                    v2.RandomHorizontalFlip(),
                    v2.SanitizeBoundingBoxes(),
                    v2.Resize(size=image_resize),
                    # Only convert the box to normalized cxcywh format in training
                    # as the eval will be done against coco original bbox
                    ConvertBox(out_fmt="cxcywh", normalize=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            target_keys=("boxes", "labels", "image_size"),
            # target_handlers={"labels": remap_labels},
        )
        self.val_dataset = CocoLikeDetection(
            root=val_data_path,
            annFile=val_ann_file,
            # The val dataset will not touch the targets
            transform=v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(size=image_resize),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            target_keys=("boxes", "labels", "image_size"),
            # target_handlers={"labels": remap_labels},
        )

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate,
        )
