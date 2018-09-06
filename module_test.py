import tensorflow as tf
import pdb
import pprint
import array
from numpy import *

#module_path = '/Users/yameng/workspace/git_repos/tensorflow/bazel-bin/tensorflow/core/user_ops/zero_out.so'
# module_path = './zero_out.so'
module_path = './mul_custom.so'

# class ZeroOutTest(tf.test.TestCase):
#     def testZeroOut(self):
#         zero_out_module = tf.load_op_library('/Users/yameng/workspace/' \
#                                              'git_repos/tensorflow/bazel-bin/tensorflow/core/user_ops/zero_out.so')
#         with self.test_session():
#             result = zero_out_module.zero_out([5, 4, 3, 2, 1])
#             self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

# if __name__ == "__main__":
#     tf.test.main()


# zero_out_module = tf.load_op_library(module_path)
# with tf.Session(''):
#   zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# # Prints
# print(array([[1, 0], [0, 0]], dtype=int32))


class MulOPTest(tf.test.TestCase):
    def testMulOP(self):
        mul_op_module = tf.load_op_library(module_path)
        # pprint.pprint(mul_op_module.__dict__)
        print("customized operation list", mul_op_module.OP_LIST)
        result = mul_op_module.mul_custom([5, 4, 3, 2, 1], [1, 2, 3, 4, 5])
        self.assertAllEqual(result.eval(), [5, 8, 9, 8, 5])

if __name__ == "__main__":
    tf.test.main()