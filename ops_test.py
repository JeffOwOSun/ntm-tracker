import numpy as np
import tensorflow as tf

from ops import batched_smooth_cosine_similarity

class BatchedSmoothCosineSimilarityTest(tf.test.TestCase):
    def testSmoothCosineSimilarity(self):
        """Test code for torch:

            th> x=torch.Tensor{{1,2,3},{2,2,2},{3,2,1},{0,2,4}}
            th> y=torch.Tensor{2,2,2}
            th> c=nn.SmoothCosineSimilarity()
            th> c:forward{x,y}
             0.9257
             0.9999
             0.9257
             0.7745
            [torch.DoubleTensor of size 4]
        """
        memory = tf.constant(
            [[[1,2,3],
             [2,2,2],
             [3,2,1],
             [0,2,4]]], dtype=np.float32)
        keys = tf.constant([[[2,2,2],[1,2,3]]], dtype=np.float32)
        for use_gpu in [True, False]:
            with self.test_session(use_gpu=use_gpu):
                loss = batched_smooth_cosine_similarity(memory, keys).eval()
                self.assertAllClose(loss,
                        [[[0.92574867671153,
                           0.99991667361053,
                           0.92574867671153,
                           0.77454667246876],
                          [0.999928,0.925749,0.714235,0.956126]]])

if __name__ == '__main__':
    tf.test.main()
