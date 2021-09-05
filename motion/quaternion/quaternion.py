"""
Adapted from pyquaternion
https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
"""

import torch
from math import pi


class Quaternion(object):
    def __init__(self, *args, **kwargs):
        s = len(args)
        if s == 0:
            if ("axis" in kwargs) and ("angle" in kwargs):
                axis = kwargs["axis"]
                angle = kwargs["angle"]
                self._q = Quaternion._from_axis_angle(axis, angle).q
            elif ("axis" in kwargs) and ("rodriguez_parameter" in kwargs):
                axis = kwargs["axis"]
                rodriguez_parameter = kwargs["rodriguez_parameter"]
                self._q = Quaternion._from_rodrigues_vector(axis, rodriguez_parameter).q
        else:
            q = args[0]
            if Quaternion.is_quaternion(q):
                self._q = q._q
            else:
                assert q.shape[-1] == 4, 'Quaternion has to be of dimension 4'
                self._q = q

    @classmethod
    def _from_axis_angle(cls, axis, angle):
        """Initialise from axis and angle representation
        Create a Quaternion by specifying the 3-vector rotation axis and rotation
        angle (in radians) from which the quaternion's rotation should be created.
        Params:
            axis: a valid numpy 3-vector
            angle: a real valued angle in radians
        """
        norm = axis.square().sum(-1).sqrt().unsqueeze(-1)
        axis = axis / norm.clamp_min(1e-12)
        theta = angle.unsqueeze(-1) / 2.0
        r = torch.where(norm > 1e-12,  torch.cos(theta), torch.ones_like(theta))
        i = torch.where(norm > 1e-12, axis * torch.sin(theta), torch.zeros_like(axis))
        q = torch.cat([r, i], dim=-1)
        return cls(q)

    @classmethod
    def _from_rodrigues_vector(cls, axis, rodrigues_parameter):
        norm = axis.square().sum(-1).sqrt().unsqueeze(-1)
        axis = axis / norm.clamp_min(1e-12)
        theta = torch.atan(rodrigues_parameter).unsqueeze(-1)
        r = torch.where(norm > 1e-12, torch.cos(theta), torch.ones_like(theta))
        i = torch.where(norm > 1e-12, axis * torch.sin(theta), torch.zeros_like(axis))
        q = torch.cat([r, i], dim=-1)
        return cls(q)

    def __mul__(self, other):
        if Quaternion.is_quaternion(other):
            return self.__class__(torch.matmul(self._q_matrix(), other._q.unsqueeze(-1)).squeeze(-1))
        return self * self.__class__(other)

    @classmethod
    def mul_(cls, q1, q2):
        return (cls(q1) * cls(q2)).q

    def __repr__(self):
        return f"Quaternion: {self._q.__repr__()}"

    def _q_matrix(self):
        """Matrix representation of quaternion for multiplication purposes.
        """
        return torch.stack([
            torch.stack([self._q[..., 0], -self._q[..., 1], -self._q[..., 2], -self._q[..., 3]], dim=-1),
            torch.stack([self._q[..., 1],  self._q[..., 0], -self._q[..., 3],  self._q[..., 2]], dim=-1),
            torch.stack([self._q[..., 2],  self._q[..., 3],  self._q[..., 0], -self._q[..., 1]], dim=-1),
            torch.stack([self._q[..., 3], -self._q[..., 2],  self._q[..., 1],  self._q[..., 0]], dim=-1)], dim=-2)

    def _q_bar_matrix(self):
        """Matrix representation of quaternion for multiplication purposes.
        """
        return torch.stack([
            torch.stack([self._q[..., 0], -self._q[..., 1], -self._q[..., 2], -self._q[..., 3]], dim=-1),
            torch.stack([self._q[..., 1],  self._q[..., 0],  self._q[..., 3], -self._q[..., 2]], dim=-1),
            torch.stack([self._q[..., 2], -self._q[..., 3],  self._q[..., 0],  self._q[..., 1]], dim=-1),
            torch.stack([self._q[..., 3],  self._q[..., 2], -self._q[..., 1],  self._q[..., 0]], dim=-1)], dim=-2)

    def _rotate_quaternion(self, q):
        """Rotate a quaternion vector using the stored rotation.

        Params:
            q: The vector to be rotated, in quaternion form (0 + xi + yj + kz)

        Returns:
            A Quaternion object representing the rotated vector in quaternion from (0 + xi + yj + kz)
        """
        #self._normalize()
        return self.__class__(self * q * self.conjugate)

    @classmethod
    def rotate_(cls, q, v):
        return cls(q).rotate(v)

    def rotate(self, vector):
        """Rotate a 3D vector by the rotation stored in the Quaternion object.

        Params:
            vector: A 3-vector specified as any ordered sequence of 3 real numbers corresponding to x, y, and z values.
                Some types that are recognised are: numpy arrays, lists and tuples.
                A 3-vector can also be represented by a Quaternion object who's scalar part is 0 and vector part is the required 3-vector.
                Thus it is possible to call `Quaternion.rotate(q)` with another quaternion object as an itorch.t.

        Returns:
            The rotated vector returned as the same type it was specified at itorch.t.

        Raises:
            TypeError: if any of the vector elements cannot be converted to a real number.
            ValueError: if `vector` cannot be interpreted as a 3-vector or a Quaternion object.

        """
        if Quaternion.is_quaternion(vector):
            return self._rotate_quaternion(vector)
        q = Quaternion(torch.cat([torch.zeros_like(vector[..., [0]]), vector], dim=-1))
        a = self._rotate_quaternion(q).vector
        return a

    @classmethod
    def conjugate_(cls, q):
        return cls(q).conjugate.q

    @property
    def conjugate(self):
        """Quaternion conjugate, encapsulated in a new instance.
        For a unit quaternion, this is the same as the inverse.
        Returns:
            A new Quaternion object clone with its vector part negated
        """
        return self.__class__(torch.cat([self.scalar.unsqueeze(-1), -self.vector], dim=-1))

    def _normalize(self):
        """Object is guaranteed to be a unit quaternion after calling this
        operation UNLESS the object is equivalent to Quaternion(0)
        """
        self._q = torch.nn.functional.normalize(self._q, dim=-1)

    @property
    def scalar(self):
        """ Return the real or scalar component of the quaternion object.

        Returns:
            A real number i.e. float
        """
        return self._q[..., 0]

    @property
    def vector(self):
        """ Return the imaginary or vector component of the quaternion object.

        Returns:
            A numpy 3-array of floats. NOT guaranteed to be a unit vector
        """
        return self._q[..., 1:]

    @classmethod
    def rotation_matrix_(cls, q):
        return cls(q).rotation_matrix

    @property
    def rotation_matrix(self):
        """Get the 3x3 rotation matrix equivalent of the quaternion rotation.

        Returns:
            A 3x3 orthogonal rotation matrix as a 3x3 Numpy array

        Note:
            This feature only makes sense when referring to a unit quaternion.
            Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.

        """
        #self._normalize()
        product_matrix = torch.matmul(self._q_matrix(), self._q_bar_matrix().conj().transpose(-2, -1))
        return product_matrix[..., 1:, 1:]

    @property
    def normalized(self):
        """Get a unit quaternion (versor) copy of this Quaternion object.

        A unit quaternion has a `norm` of 1.0

        Returns:
            A new Quaternion object clone that is guaranteed to be a unit quaternion
        """
        q = Quaternion(self._q)
        q._normalize()
        return q

    @property
    def q(self):
        return self._q

    @classmethod
    def euler_angle_(cls, q, order, epsilon=0.):
        return cls(q).euler_angle(order, epsilon)

    def euler_angle(self, order: str = 'zyx', epsilon: float = 0.):
        """
        Convert quaternion(s) q to Euler angles.
        """
        assert self.q.shape[-1] == 4

        original_shape = list(self.q.shape)
        original_shape[-1] = 3
        q = self.q.view(-1, 4)

        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]

        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        else:
            raise

        return torch.stack((x, y, z), dim=1).view(original_shape)

    def get_axis(self):
        """Get the axis or vector about which the quaternion rotation occurs
        For a null rotation (a purely real quaternion), the rotation angle will
        always be `0`, but the rotation axis is undefined.
        It is by default assumed to be `[0, 0, 0]`.
        Params:
            undefined: [optional] specify the axis vector that should define a null rotation.
                This is geometrically meaningless, and could be any of an infinite set of vectors,
                but can be specified if the default (`[0, 0, 0]`) causes undesired behaviour.
        Returns:
            A Numpy unit 3-vector describing the Quaternion object's axis of rotation.
        Note:
            This feature only makes sense when referring to a unit quaternion.
            Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """
        tolerance = 1e-12
        #self._normalize()
        norm = torch.norm(self.vector, dim=-1).unsqueeze(-1)#self.vector.square().sum(-1).sqrt().unsqueeze(-1)
        return torch.where(norm > tolerance, torch.nn.functional.normalize(self.vector, dim=-1), torch.zeros_like(self.vector))

    @property
    def axis(self):
        return self.get_axis()

    def _wrap_angle(self, theta):
        """Helper method: Wrap any angle to lie between -pi and pi
        Odd multiples of pi are wrapped to +pi (as opposed to -pi)
        """
        return torch.remainder(theta + pi, 2 * pi) - pi

    @property
    def angle(self):
        """Get the angle (in radians) describing the magnitude of the quaternion rotation about its rotation axis.
        This is guaranteed to be within the range (-pi:pi) with the direction of
        rotation indicated by the sign.
        When a particular rotation describes a 180 degree rotation about an arbitrary
        axis vector `v`, the conversion to axis / angle representation may jump
        discontinuously between all permutations of `(-pi, pi)` and `(-v, v)`,
        each being geometrically equivalent (see Note in documentation).
        Returns:
            A real number in the range (-pi:pi) describing the angle of rotation
                in radians about a Quaternion object's axis of rotation.
        Note:
            This feature only makes sense when referring to a unit quaternion.
            Calling this method will implicitly normalise the Quaternion object to a unit quaternion if it is not already one.
        """
        #self._normalize()
        norm = torch.norm(self.vector, dim=-1)
        return self._wrap_angle(2.0 * torch.atan2(norm, self.scalar))

    @property
    def axis_angle(self):
        return self.angle.unsqueeze(-1) * self.axis

    @property
    def rodriguez_vector(self):
        return torch.tan(self.angle.unsqueeze(-1) / 2.) * self.axis

    @staticmethod
    def is_quaternion(other):
        return 'Quaternion' in other.__class__.__name__

    @staticmethod
    def qfix_(q: torch.Tensor) -> torch.Tensor:
        """
        Enforce quaternion continuity across the time dimension by selecting
        the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
        between two consecutive frames.

        Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
        Returns a tensor of the same shape.
        """
        assert len(q.shape) == 3
        assert q.shape[-1] == 4

        result = q.clone()
        dot_products = torch.sum(q[1:] * q[:-1], dim=2)
        mask = dot_products < 0
        mask = (torch.cumsum(mask, dim=0) % 2).type(torch.bool)
        result[1:][mask] *= -1
        return result

    @staticmethod
    def qfix_pos_(q: torch.Tensor) -> torch.Tensor:
        """
        Enforce quaternion w to be positive

        Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
        Returns a tensor of the same shape.
        """
        assert len(q.shape) == 3
        assert q.shape[-1] == 4

        mask = q[..., 0] < 0.
        q_out = q.clone()
        q_out[mask] *= -1.
        return q_out
