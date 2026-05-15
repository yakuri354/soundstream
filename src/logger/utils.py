import io

import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_spectrogram(spectrogram, name=""):
    """
    Plot spectrogram

    Args:
        spectrogram (Tensor): spectrogram tensor.
        name (None | str): optional name.
    Returns:
        image (Image): image of the spectrogram
    """
    plt.figure(figsize=(20, 5))
    plt.pcolormesh(spectrogram)
    plt.title(name)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # convert buffer to Tensor
    image = ToTensor()(PILImage.open(buf))

    plt.close()

    return image
