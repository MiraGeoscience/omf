"""texture.py: contains ImageTexture definition"""

import properties

from .base import ContentModel
from .serializers import png_deserializer, png_serializer


class ImageTexture(ContentModel):
    """Contains an image that can be mapped to a point set or surface"""

    origin = properties.Vector3("Origin point of the texture", default=[0.0, 0.0, 0.0])
    axis_u = properties.Vector3("Vector corresponding to the image x-axis", default="X")
    axis_v = properties.Vector3("Vector corresponding to the image y-axis", default="Y")
    image = properties.ImagePNG(
        "PNG image file", serializer=png_serializer, deserializer=png_deserializer
    )
