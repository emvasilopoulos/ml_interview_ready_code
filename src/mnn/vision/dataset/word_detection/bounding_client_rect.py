import dataclasses


@dataclasses.dataclass
class BoundingClientRect:
    x: float
    y: float
    width: float
    height: float
    top: float
    right: float
    bottom: float
    left: float

    @staticmethod
    def from_dict(dictionary: dict):
        return BoundingClientRect(
            x=dictionary["x"],
            y=dictionary["y"],
            width=dictionary["width"],
            height=dictionary["height"],
            top=dictionary["top"],
            right=dictionary["right"],
            bottom=dictionary["bottom"],
            left=dictionary["left"],
        )
