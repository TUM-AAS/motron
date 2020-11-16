import torch
from common.quaternion import qmat, matq, ortho6dmat
from motion.state_representation import StateRepresentation


class Ortho6DRepresentation(StateRepresentation):
    @staticmethod
    def sum(v1, v2):
        assert v1.shape[-1] == 3 and v1.shape[-2] >= 2
        assert v1.shape[-1] == v2.shape[-1] and v1.shape[-2] == v2.shape[-2]
        return torch.matmul(ortho6dmat(v2), ortho6dmat(v1))[..., :-1, :]

    @staticmethod
    def size():
        return 6

    @staticmethod
    def convert_from_quaternions(q):
        return qmat(q)[..., :-1, :]

    @staticmethod
    def validate(v):
        assert v.shape[-1] == 3 and v.shape[-2] >= 2
        return ortho6dmat(v)[..., :-1, :]

    @staticmethod
    def convert_to_quaternions(r):
        return matq(ortho6dmat(r))
