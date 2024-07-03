import dataclasses
from typing import List

import mnn.vision.dataset.word_detection.bounding_client_rect as bounding_client_rect


@dataclasses.dataclass
class RawDatasetImageShape:
    width: int
    height: int

    @staticmethod
    def from_dict(data: dict):
        return RawDatasetImageShape(width=data["width"], height=data["height"])


@dataclasses.dataclass
class RawDatasetRenderedElement:
    nodeName: str
    boundingClientRect: bounding_client_rect.BoundingClientRect
    innerText: str
    className: str

    @staticmethod
    def from_dict(data: dict):
        return RawDatasetRenderedElement(
            nodeName=data["nodeName"],
            boundingClientRect=bounding_client_rect.BoundingClientRect.from_dict(
                data["boundingClientRect"]
            ),
            innerText=data["innerText"],
            className=data["className"] if "className" in data else None,
        )


@dataclasses.dataclass
class RawDataset:
    body_shape: RawDatasetImageShape
    rendered_elements: List[RawDatasetRenderedElement]

    @staticmethod
    def from_dict(data: dict):
        return RawDataset(
            body_shape=RawDatasetImageShape.from_dict(data["body_shape"]),
            rendered_elements=[
                RawDatasetRenderedElement.from_dict(element)
                for element in data["rendered_elements"]
            ],
        )
