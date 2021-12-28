import paddle
from typing import Sequence, Union


Size = Union[Sequence[int], int]


def swap_axes(src: paddle.Tensor, axes_1: Size, axes_2: Size):
    if isinstance(axes_1, int):
        axes_1 = [axes_1]
    if isinstance(axes_2, int):
        axes_2 = [axes_2]
    dims = [*range(src.dim())]
    for a, b in zip(axes_1, axes_2):
        dims[a], dims[b] = dims[b], dims[a]
    return src.transpose(dims)


def repeat_interleave(src: paddle.Tensor, repeat: Union[paddle.Tensor, int], axis=None):
    if axis is None:
        src = src.reshape([-1])
        axis = 0
    dtype = 'int64'
    if isinstance(repeat, paddle.Tensor):
        dtype = repeat.dtype
    repeat = repeat + paddle.zeros([src.shape[axis]], dtype=dtype)
    cumptr = paddle.cumsum(repeat)
    src_sw = swap_axes(src, 0, axis)
    return swap_axes(cum_repeat_0(src_sw, cumptr), 0, axis)


def cum_repeat_0(src: paddle.Tensor, cumptr: paddle.Tensor):
    idx = paddle.scatter(
        paddle.zeros([cumptr[-1]], dtype=cumptr.dtype),
        cumptr, paddle.ones_like(cumptr),
        overwrite=False
    )
    visit = paddle.cumsum(idx)
    return paddle.gather(src, visit, axis=0)
