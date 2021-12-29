# Sparse-dense operators implementation for Paddle
This module implements `coo`, `csc` and `csr` matrix formats and their inter-ops with dense matrices.

## Requirements
It only needs `paddle`. It is tested on `paddle >= 2.1.0, <= 2.2.0rc1`, but should work for any recent paddle versions.

## Usage
Most functions are implemented within classes that encapsulate sparse formats: `COO`, `CSR` and `CSC`.

Cross-format operators are implemented in dedicated sub-modules: `spgemm`.

## Supported operations

### Conversion
```plain
coo -> csc, csr, dense
csc -> coo
csr -> coo
```

### Batch MVP (Matrix-Vector Product) or SpMM (Sparse-Dense Matmul)
Note that in this library, the batch dimensions are appended instead of prepended to the dot dimension (which makes batch MVP essentially regular matmul). Use `utils.swap_axes` or `paddle.transpose` when necessary.
```plain
coo, dense -> dense
```

### Point-wise
Supports broadcast on the dense side.
```plain
coo + coo -> coo
coo * scalar -> coo
coo * dense -> coo (equiv. coo @ diag(vec) if dense is a vector)
```

### SpGEMM (Sparse-Sparse Matmul)
```plain
coo, csr -> coo (via row-wise mixed product)
```

## Installation
The project doesn't have any stable release yet. You may install it from this git repository directly.

## Caveats
Currently all stuff is implemented with pure python and no CUDA code has been written. As a result, the routines have good run-time performance in general but have a memory overhead of order `O(nnz/n)`.
