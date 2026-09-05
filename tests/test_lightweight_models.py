import unittest

import torch

from model_zoo.lightweight_blocks import DepthwiseSeparableConv
from model_zoo.registry import MODEL_REGISTRY, create_model


class LightweightModelForwardTest(unittest.TestCase):
    def test_depthwise_separable_contract(self):
        block = DepthwiseSeparableConv(4, 12)
        self.assertEqual(block.depthwise.groups, 4)
        self.assertEqual(block.depthwise.kernel_size, (3, 3))
        self.assertEqual(block.pointwise.kernel_size, (1, 1))
        self.assertEqual(block(torch.randn(1, 4, 17, 19)).shape, (1, 12, 17, 19))

    def test_all_models_preserve_four_channel_raw_shape(self):
        for name in MODEL_REGISTRY:
            with self.subTest(model=name):
                model = create_model(name).eval()
                inputs = torch.randn(1, 4, 65, 97)
                with torch.inference_mode():
                    outputs = model(inputs)
                self.assertEqual(outputs.shape, inputs.shape)
                self.assertEqual(outputs.shape[1], 4)


if __name__ == "__main__":
    unittest.main()
