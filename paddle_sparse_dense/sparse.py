from typing import Sequence
import paddle
from . import utils


class COO:
    def __init__(
        self,
        shape: Sequence[int],
        row: paddle.Tensor, col: paddle.Tensor, data: paddle.Tensor
    ) -> None:
        assert len(shape) == 2, ["COO: not a matrix; expected 2d shape, found", shape]
        assert \
            row.dim() == col.dim() == data.dim() == 1, \
            [
                "Invalid COO data, expected 1d data but got dims for R/C/D as",
                row.dim(), col.dim(), data.dim()
            ]
        assert len(row) == len(col) == len(data), [
            "Invalid COO data, different lengths for R/C/D",
            len(row), len(col), len(data)
        ]
        self.row = row
        self.col = col
        self.data = data
        self.shape = shape

    def scale(self, s):
        """
        Scalar multiplication.
        """
        return COO(self.row, self.col, self.data * s)

    def mul(self, ov: paddle.Tensor):
        """
        Element-wise multiplication. Supports broadcast on v.
        Equivalent to self.dot(diag(v)) if v is a vector.
        [m, k], [m or 1, k or 1] or [k] -> [m, k]
        """
        v = ov
        if v.dim() == 1:
            v = v.reshape([1, -1])
        assert v.dim() == 2, [
            "COO.mul requires at least 1-dim and at most 2-dims for operand, got shape",
            v.shape
        ]
        assert self.shape[0] == v.shape[0] or v.shape[0] == 1, [
            "COO mul shape mismatch", self.shape, ov.shape
        ]
        assert self.shape[1] == v.shape[1] or v.shape[1] == 1, [
            "COO mul shape mismatch", self.shape, ov.shape
        ]
        i = self.row
        if v.shape[0] == 1:
            i = paddle.zeros_like(i)
        j = self.col
        if v.shape[1] == 1:
            j = paddle.zeros_like(j)
        g = paddle.gather_nd(v, paddle.stack([i, j], axis=-1))
        return COO(self.shape, self.row, self.col, self.data * g)

    def add(self, coo: 'COO'):
        assert self.shape[0] == coo.shape[0], ["COO add shape mismatch", self.shape, coo.shape]
        assert self.shape[1] == coo.shape[1], ["COO add shape mismatch", self.shape, coo.shape]
        return COO(
            self.shape,
            paddle.concat([self.row, coo.row]),
            paddle.concat([self.col, coo.col]),
            paddle.concat([self.data, coo.data])
        )

    def dot(self, v: paddle.Tensor):
        """
        Dot product on the second dim and v's first dim.
        [m, k], [k, n_1, n_2, ...] -> [m, n_1, n_2, ...]
        """
        assert self.shape[1] == len(v), ["Shape mismatch: ", self.shape, len(v)]
        result = paddle.zeros([self.shape[0]] + [*v.shape[1:]], dtype=v.dtype)
        return paddle.scatter(
            result, self.row, paddle.gather(v, self.col, axis=0) * self.data.reshape(
                [-1] + [1] * (len(v.shape) - 1)
            ), overwrite=False
        )

    def csr(self):
        rcn = paddle.scatter(
            paddle.zeros([self.shape[0]], dtype=self.row.dtype),
            self.row,
            paddle.ones_like(self.row),
            overwrite=False
        )
        ras = paddle.argsort(self.row)
        nidx = paddle.gather(self.col, ras, axis=0)
        ndata = paddle.gather(self.data, ras, axis=0)
        return CSR(
            self.shape, paddle.cumsum(rcn), nidx, ndata
        )

    def csc(self):
        ccn = paddle.scatter(
            paddle.zeros([self.shape[0]], dtype=self.col.dtype),
            self.col,
            paddle.ones_like(self.col),
            overwrite=False
        )
        cas = paddle.argsort(self.col)
        nidx = paddle.gather(self.row, cas, axis=0)
        ndata = paddle.gather(self.data, cas, axis=0)
        return CSC(
            self.shape, paddle.cumsum(ccn), nidx, ndata
        )

    def dense(self):
        return paddle.scatter_nd(
            paddle.stack([self.row, self.col], axis=-1),
            self.data, self.shape
        )


class CSR:
    def __init__(self, shape, ptr, idx, data) -> None:
        assert len(shape) == 2, ["CSR: not a matrix; expected 2d shape, found", shape]
        assert \
            ptr.dim() == idx.dim() == data.dim() == 1, \
            [
                "Invalid CSR data, expected 1d data but got dims for P/I/D as",
                ptr.dim(), idx.dim(), data.dim()
            ]
        assert len(idx) == len(data), [
            "Invalid CSR data, different lengths for I/D",
            len(idx), len(data)
        ]
        assert len(ptr) == shape[0], ["Invalid CSR data, P/S0", len(ptr), shape[0]]
        self.ptr = ptr
        self.idx = idx
        self.data = data
        self.shape = shape

    def coo(self):
        rows = paddle.arange(len(self.ptr), dtype=self.ptr.dtype)
        rows = utils.cum_repeat_0(rows, self.ptr)
        return COO(self.shape, rows, self.idx, self.data)


class CSC:
    def __init__(self, shape, ptc, idx, data) -> None:
        assert len(shape) == 2, ["CSC: not a matrix; expected 2d shape, found", shape]
        assert \
            ptc.dim() == idx.dim() == data.dim() == 1, \
            [
                "Invalid CSC data, expected 1d data but got dims for P/I/D as",
                ptc.dim(), idx.dim(), data.dim()
            ]
        assert len(idx) == len(data), [
            "Invalid CSC data, different lengths for I/D",
            len(idx), len(data)
        ]
        assert len(ptc) == shape[0], ["Invalid CSC data, P/S0", len(ptc), shape[0]]
        self.ptc = ptc
        self.idx = idx
        self.data = data
        self.shape = shape

    def coo(self):
        cols = paddle.arange(len(self.ptc), dtype=self.ptc.dtype)
        cols = utils.cum_repeat_0(cols, self.ptc)
        return COO(self.shape, self.idx, cols, self.data)
