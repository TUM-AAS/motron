import torch

from common.quaternion import qmul, qpositive, qeuler, eulerq
from motion.state_representation import StateRepresentation


class EulerAngleRepresentation(StateRepresentation):
    @staticmethod
    def sum(v1, v2):
        assert v1.shape[-1] == 3
        assert v1.shape[-1] == v2.shape[-1]
        return v1 + v2

    @staticmethod
    def size():
        return 3

    @staticmethod
    def validate(v):
        assert v.shape[-1] == 3
        return v

    @staticmethod
    def convert_from_quaternions(q):
        return qeuler(q, order='zyx', epsilon=1e-6)

    @staticmethod
    def convert_to_quaternions(q):
        return eulerq(q, order='zyx')
