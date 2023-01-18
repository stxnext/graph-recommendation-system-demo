import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import SAGEConv

from recommendations import DEVICE

from recommendations.consts import ENCODER_MODEL_NAME


class SequenceEncoder(object):
    """
    The 'SequenceEncoder' encodes raw column strings into embeddings.
    """
    def __init__(self, model_name: str = ENCODER_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name, device=DEVICE)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=DEVICE)
        return x.cpu()

class LabelsEncoder(object):
    """
    The 'LabelsEncoder' splits the raw column strings by 'sep' and converts
    individual elements to categorical labels.
    """
    def __init__(self, sep: str = '|') -> None:
        self.sep = sep

    def __call__(self, df: pd.DataFrame) -> torch.tensor:
        labels = set(g for col in df.values for g in col.split(self.sep))
        mapping = {label: i for i, label in enumerate(labels)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x

class IdentityEncoder(object):
    """
    The 'IdentityEncoder' takes the raw column values and converts them to
    PyTorch tensors.
    """
    def __init__(self, dtype: torch.dtype = None, is_list: bool = False) -> None:
        self.dtype = dtype
        self.is_list = is_list

    def __call__(self, df: pd.DataFrame) -> torch.tensor:
        if self.is_list:
            return torch.stack([torch.tensor(el) for el in df.values])
        return torch.from_numpy(df.values).to(self.dtype)

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


