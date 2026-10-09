"""Experimental frozen-column growth primitives for isolated PyTorch pilots.

These change architecture and cannot be exported as an unchanged native Gemma.
Old routes remain exact only when callers keep the old route and frozen state.
"""


def residual_column(predecessor, width: int, layers: int = 1, bottleneck: int = 16):
    import copy

    import torch
    from torch import nn

    if any(type(n) is not int for n in (width, layers, bottleneck)) or not (
        1 <= layers <= 4 and 1 <= bottleneck <= width <= 4096
    ):
        raise ValueError("Growth column dimensions exceed the pilot budget")
    if layers * width * bottleneck * 2 > 2_000_000:
        raise ValueError("Growth column exceeds two million new parameters")
    if sum(p.numel() for p in predecessor.parameters()) > 2_000_000:
        raise ValueError("Frozen-column primitives are bounded tiny-model pilots")
    predecessor = copy.deepcopy(predecessor)

    class FrozenColumn(nn.Module):
        def __init__(self):
            super().__init__()
            self.predecessor = predecessor
            self.predecessor.requires_grad_(False)
            self.predecessor.eval()
            self.blocks = nn.ModuleList()
            parameter = next(predecessor.parameters())
            for _ in range(layers):
                block = nn.Sequential(
                    nn.Linear(width, bottleneck), nn.GELU(), nn.Linear(bottleneck, width)
                )
                block.to(device=parameter.device, dtype=parameter.dtype)
                nn.init.zeros_(block[-1].weight)
                nn.init.zeros_(block[-1].bias)
                self.blocks.append(block)

        def train(self, mode=True):
            super().train(mode)
            self.predecessor.eval()
            return self

        def forward(self, value, route="previous"):
            if route not in ("previous", "expanded"):
                raise ValueError("Unknown growth route")
            with torch.no_grad():
                value = self.predecessor(value)
            if route == "expanded":
                for block in self.blocks:
                    value = value + block(value)
            return value

    return FrozenColumn()


def embedding_extension(predecessor, new_rows: int):
    """Freeze original IDs; append trainable rows without renumbering any token.

    Requires a matching tokenizer AND output head in an architecture pilot.
    This primitive alone does not add vocabulary to the serving language model.
    """
    import torch
    from torch import nn

    if type(new_rows) is not int or not 1 <= new_rows <= 4096:
        raise ValueError("Embedding extension needs 1..4096 rows")
    if (
        not isinstance(predecessor, nn.Embedding)
        or new_rows * predecessor.embedding_dim > 2_000_000
        or predecessor.weight.numel() > 2_000_000
    ):
        raise ValueError("Embedding extension exceeds its parameter budget")

    class ExpandedEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("original", predecessor.weight.detach().clone())
            self.extra = nn.Embedding(new_rows, predecessor.embedding_dim).to(
                device=predecessor.weight.device, dtype=predecessor.weight.dtype
            )
            self.extra.weight.data.copy_(self.original.mean(dim=0).expand_as(self.extra.weight))

        def forward(self, ids):
            if (
                ids.dtype not in (torch.int32, torch.int64)
                or (ids < 0).any()
                or (ids >= len(self.original) + new_rows).any()
            ):
                raise ValueError("Embedding token ID outside expanded vocabulary")
            table = torch.cat((self.original, self.extra.weight), dim=0)
            return torch.nn.functional.embedding(ids, table)

    return ExpandedEmbedding()
