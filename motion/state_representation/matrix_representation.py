import torch

from common.quaternion import qmat, matq
from motion.state_representation import StateRepresentation


class MatrixRepresentation(StateRepresentation):
    @staticmethod
    def sum(v1, v2):
        assert v1.shape[-1] == 3 and v1.shape[-2] == 3
        assert v1.shape[-1] == v2.shape[-1] and v1.shape[-2] == v2.shape[-2]
        return torch.matmul(v2, v1)

    @staticmethod
    def size():
        return 9

    @staticmethod
    def convert_from_quaternions(q):
        return qmat(q)

    @staticmethod
    def convert_to_quaternions(rot):
        return matq(rot)
