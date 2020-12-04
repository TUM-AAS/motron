import torch

from common.quaternion import expmap_to_quaternion, qfix
from motion.state_representation import StateRepresentation


class ExponentialMapRepresentation(StateRepresentation):
    @staticmethod
    def sum(v1, v2):
        raise NotImplementedError

    @staticmethod
    def size():
        return 3

    @staticmethod
    def validate(v):
        assert v.shape[-1] == 3
        return v

    @staticmethod
    def convert_from_quaternions(q):
        return q

    @staticmethod
    def convert_to_quaternions(x):
        quat = expmap_to_quaternion(-x)
        quat = qfix(quat)
        return quat
