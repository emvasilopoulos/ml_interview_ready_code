import dataclasses

import mnn.vision.dataset.word_detection.bounding_client_rect as bounding_client_rect


@dataclasses.dataclass
class WordDetectionLabel:
    class_id: int
    x: float
    y: float
    width: float
    height: float


@dataclasses.dataclass
class WordDetectionLabelWithText(WordDetectionLabel):
    text: str


def _get_bbox(
    bounding_client_rect: bounding_client_rect.BoundingClientRect,
    normalized: bool,
    image_width: int,
    image_height: int,
):
    if normalized:
        x = bounding_client_rect.x / image_width
        y = bounding_client_rect.y / image_height
        width = bounding_client_rect.width / image_width
        height = bounding_client_rect.height / image_height
    else:
        x = bounding_client_rect.x
        y = bounding_client_rect.y
        width = bounding_client_rect.width
        height = bounding_client_rect.height
    return x, y, width, height


def get_label_from_bounding_client_rect(
    class_id: int,
    bounding_client_rect: bounding_client_rect.BoundingClientRect,
    normalized: bool,
    image_width: int,
    image_height: int,
) -> WordDetectionLabel:
    x, y, width, height = _get_bbox(
        bounding_client_rect, normalized, image_width, image_height
    )
    return WordDetectionLabel(class_id, x, y, width, height)


def get_label_with_text_from_bounding_client_rect(
    class_id: int,
    bounding_client_rect: bounding_client_rect.BoundingClientRect,
    normalized: bool,
    image_width: int,
    image_height: int,
    text: str,
) -> WordDetectionLabelWithText:
    x, y, width, height = _get_bbox(
        bounding_client_rect, normalized, image_width, image_height
    )
    return WordDetectionLabelWithText(class_id, x, y, width, height, text)
