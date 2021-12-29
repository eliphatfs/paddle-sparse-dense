import paddle_sparse_dense
import paddle
import numpy
import scipy.sparse as sp
import unittest
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TestUtils(unittest.TestCase):
    def test_repeat_interleave_simple(self):
        self.assertListEqual(paddle_sparse_dense.utils.repeat_interleave(
            paddle.to_tensor(numpy.array([1, 2, 3])),
            paddle.to_tensor(numpy.array([3, 0, 1]))
        ).tolist(), [1, 1, 1, 3])

    def test_repeat_interleave(self):
        for _ in range(10):
            src = numpy.random.randn(19, 11)
            gt1 = src.repeat(2, axis=0)
            pr1 = paddle_sparse_dense.utils.repeat_interleave(
                paddle.Tensor(src),
                2, axis=0
            )
            rep = numpy.random.randint(4, size=[11])
            gt2 = numpy.repeat(src, rep, axis=1)
            pr2 = paddle_sparse_dense.utils.repeat_interleave(
                paddle.Tensor(src),
                paddle.Tensor(rep), axis=1
            )
            gt3 = src.repeat(3)
            pr3 = paddle_sparse_dense.utils.repeat_interleave(
                paddle.Tensor(src), 3
            )
            self.assertTrue(numpy.allclose(gt1, pr1))
            self.assertTrue(numpy.allclose(gt3, pr3))
            self.assertTrue(numpy.allclose(gt2, pr2))


class TestSparseDense(unittest.TestCase):
    def random_coo_gen(self):
        for i in range(2, 19):
            n = int(1.4 ** i)
            k = int(1.3 ** i)
            nnz = int(1.6 ** i)
            r = numpy.random.randint(n, size=[nnz])
            c = numpy.random.randint(k, size=[nnz])
            d = numpy.random.randn(nnz)
            yield r, c, d, n, k

    def random_coo_pair_gen(self):
        for i in range(2, 19):
            n = int(1.4 ** i)
            k = int(1.3 ** i)
            m = int(1.2 ** i)
            nnz = int(1.6 ** i)
            nnz2 = int(1.5 ** i)
            r1 = numpy.random.randint(n, size=[nnz])
            c1 = numpy.random.randint(k, size=[nnz])
            d1 = numpy.random.randn(nnz)
            r2 = numpy.random.randint(k, size=[nnz2])
            c2 = numpy.random.randint(m, size=[nnz2])
            d2 = numpy.random.randn(nnz2)
            yield r1, c1, d1, r2, c2, d2, n, k, m

    def test_coo_dot(self):
        for r, c, d, n, k in self.random_coo_gen():
            v1 = numpy.random.randn(k)
            v2 = numpy.random.randn(k, 4)
            A = sp.coo_matrix((d, (r, c)), shape=[n, k]).todense()
            B = self.coo_from_npdata([n, k], r, c, d)
            for i, v in enumerate([v1, v2]):
                gt = A.dot(v)
                pr = B.dot(paddle.Tensor(v))
                self.assertTrue(numpy.allclose(gt, pr), "%d" % i)

    def test_conversion_dot(self):
        for r, c, d, n, k in self.random_coo_gen():
            v1 = numpy.random.randn(k)
            v2 = numpy.random.randn(k, 4)
            A = sp.coo_matrix((d, (r, c)), shape=[n, k]).todense()
            B = self.coo_from_npdata([n, k], r, c, d)
            for i, v in enumerate([v1, v2]):
                gt = A.dot(v)
                pr = B.dot(paddle.Tensor(v))
                self.assertTrue(numpy.allclose(gt, pr), "COO %d" % i)
                B = B.csr().coo()
                pr = B.dot(paddle.Tensor(v))
                self.assertTrue(numpy.allclose(gt, pr), "CSR %d" % i)
                B = B.csc().coo()
                pr = B.dot(paddle.Tensor(v))
                self.assertTrue(numpy.allclose(gt, pr), "CSC %d" % i)

    def coo_from_npdata(self, shape, r, c, d):
        return paddle_sparse_dense.COO(shape,
            paddle.Tensor(r), paddle.Tensor(c), paddle.Tensor(d)
        )

    def test_spgemm_coo_csr(self):
        for r1, c1, d1, r2, c2, d2, n, k, m in self.random_coo_pair_gen():
            A = sp.coo_matrix((d1, (r1, c1)), shape=[n, k]).todense()
            B = sp.coo_matrix((d2, (r2, c2)), shape=[k, m]).todense()
            Ap = self.coo_from_npdata([n, k], r1, c1, d1)
            Bp = self.coo_from_npdata([k, m], r2, c2, d2)
            gt = numpy.matmul(A, B)
            pr = paddle_sparse_dense.spgemm.spgemm_rowmp_coo_csr_coo(
                Ap, Bp.csr()
            ).dense()
            self.assertTrue(numpy.allclose(gt, pr), "n, k, m = %d, %d, %d" % (n, k, m))


if __name__ == "__main__":
    unittest.main()
