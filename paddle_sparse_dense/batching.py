from typing import Sequence, Tuple
import paddle
import numpy
from .sparse import COO
import bisect
import itertools


class BatchingInfo:
    def __init__(self, shapes: numpy.ndarray) -> None:
        self.shapes = shapes


def batch_info_dot(a: BatchingInfo, b: BatchingInfo):
    assert numpy.all(a.shapes[:, 1] == b.shapes[:, 0])
    return BatchingInfo(numpy.stack([a.shapes[:, 0], b.shapes[:, 1]], -1))


def batch(matrices: Sequence[COO]) -> Tuple[COO, BatchingInfo]:
    """
    Batches a sequence of COO matrices into a large COO matrix.
    This routine also returns a BatchingInfo object for use in unbatching.
    """
    s0 = numpy.cumsum([0] + [m.shape[0] for m in matrices])
    s1 = numpy.cumsum([0] + [m.shape[1] for m in matrices])
    rs = []
    cs = []
    ds = []
    for ro, co, coo in zip(s0, s1, matrices):
        rs.append(ro + coo.row)
        cs.append(co + coo.col)
        ds.append(coo.data)
    return COO(
        [s0[-1], s1[-1]], paddle.concat(rs), paddle.concat(cs), paddle.concat(ds)
    ), BatchingInfo(numpy.array([[*m.shape] for m in matrices]))


def unbatch(matrix: COO, info: BatchingInfo) -> Sequence[COO]:
    """
    Unbatches a large COO matrix into a sequence of COO matrices. 
    info: the BatchingInfo returned by `batch`.
    """
    ms = []
    ras = paddle.argsort(matrix.row)
    sr = paddle.gather(matrix.row, ras, 0)
    sc = paddle.gather(matrix.col, ras, 0)
    sd = paddle.gather(matrix.data, ras, 0)
    ofs = numpy.cumsum(info.shapes[:, 0])
    ofsr = itertools.chain([0], ofs)
    ofsc = itertools.chain([0], numpy.cumsum(info.shapes[:, 1]))
    begin = 0
    for seg_end, br, bc, shape in zip(ofs, ofsr, ofsc, info.shapes):
        seg_end = int(seg_end)
        bisect_end = bisect.bisect_left(sr, seg_end, begin)
        if begin == bisect_end:
            z = paddle.to_tensor([0], dtype=sr.dtype)
            v = paddle.to_tensor([0], dtype=sd.dtype)
            ms.append(COO(shape.tolist(), z, z, v))
        else:
            ms.append(COO(
                shape.tolist(),
                sr[begin: bisect_end] - br,
                sc[begin: bisect_end] - bc,
                sd[begin: bisect_end]
            ))
        begin = bisect_end
    return ms
