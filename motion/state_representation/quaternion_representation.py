import torch

from common.quaternion import qmul, qpositive
from motion.state_representation import StateRepresentation


class QuaternionRepresentation(StateRepresentation):
    @staticmethod
    def sum(v1, v2):
        assert v1.shape[-1] == 4
        assert v1.shape[-1] == v2.shape[-1]
        return qmul(v2, v1)

    @staticmethod
    def size():
        return 4

    @staticmethod
    def validate(v):
        assert v.shape[-1] == 4
        return torch.nn.functional.normalize(v, dim=-1)

    @staticmethod
    def convert_from_quaternions(q):
        return q
        return qpositive(q)

    @staticmethod
    def convert_to_quaternions(q):
        return q
