import paddle
from .sparse import COO, CSR
from .utils import repeat_interleave


def spgemm_rowmp_coo_csr_coo(coo: COO, csr: CSR) -> COO:
    """
    SpGEMM (sparse-sparse matmul) via row-wise mixed product. COO, CSR -> COO
    """
    assert coo.shape[1] == csr.shape[0], ["Mismatching matmul shapes", coo.shape, csr.shape]
    ext_ptr = paddle.concat([paddle.zeros_like(csr.ptr[:1]), csr.ptr])
    rowend = paddle.gather(ext_ptr, coo.col + 1, axis=0)
    rowstart = paddle.gather(ext_ptr, coo.col, axis=0)
    rowsize = rowend - rowstart
    if rowsize.sum() == 0:
        return COO(
            [coo.shape[0], csr.shape[-1]],
            paddle.zeros_like(csr.ptr[:1]),
            paddle.zeros_like(csr.ptr[:1]),
            paddle.zeros_like(csr.data[:1]),
        )
    scatter_i = repeat_interleave(coo.row, rowsize)
    msg_a = repeat_interleave(coo.data, rowsize)
    mat_b = paddle.zeros([len(msg_a) + 1], dtype=rowstart.dtype)
    scatter_k = repeat_interleave(rowstart, rowsize)
    ofs_b = paddle.cumsum(1 + paddle.scatter(
        paddle.zeros_like(mat_b), paddle.cumsum(rowsize, axis=0),
        -rowsize,
        overwrite=False
    ), axis=0)[:-1] - 1
    ofs_m = scatter_k + ofs_b
    scatter_j = paddle.gather(csr.idx, ofs_m, axis=0)
    msg_b = paddle.gather(csr.data, ofs_m, axis=0)
    return COO(
        [coo.shape[0], csr.shape[-1]],
        scatter_i, scatter_j, msg_a * msg_b
    )
