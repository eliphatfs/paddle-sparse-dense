# Sparse-dense operators implementation for Paddle
This module implements `coo`, `csc` and `csr` matrix formats and their inter-ops with dense matrices.

## Requirements
It only needs `paddle`. It is tested on `paddle >= 2.1.0, <= 2.2.0rc1`, but should work for any recent paddle versions.

## Supported operations

### Conversion
```plain
coo -> csc, csr
csc -> coo
csr -> coo
```

### Batch MVP (Matrix-Vector Product) or SpMM (Sparse-Dense Matmul)
Note that in this library, the batch dimensions are appended instead of prepended to the dot dimension (which is just regular matmul when there is one batch dimension). Use `utils.swap_axes` or `paddle.transpose` when necessary.
```plain
coo, dense -> dense
```

### Point-wise
TBD

### Sp-Sp Matmul with dense outputs
TBD

## Installation
No release version yet. You may install from this git repository.
