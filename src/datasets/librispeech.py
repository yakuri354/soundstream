from torch.utils.data import Dataset
from torchaudio.datasets.librispeech import LIBRISPEECH


class LibrispeechDataset(Dataset):
    def __init__(
        self,
        root: str,
        instance_transforms,
        sample_rate: int,
        split: str = "train-clean-100",
        limit: int | None = None,
        download: bool = True,
        base=LIBRISPEECH,
    ) -> None:
        super().__init__()
        self.ds = base(root, url=split, download=download)
        self.instance_transforms = instance_transforms
        self.sample_rate = sample_rate
        self.limit = limit

    def preprocess(self, data):
        if self.instance_transforms is not None:
            for key, tf in self.instance_transforms.items():
                data[key] = tf(data[key])

    def __getitem__(self, index):
        data, sr = self.ds[index][:2]
        assert sr == self.sample_rate

        data = {"input": data}

        self.preprocess(data)

        return data

    def __len__(self) -> int:
        if self.limit is not None:
            return min(len(self.ds), self.limit)

        return len(self.ds)
