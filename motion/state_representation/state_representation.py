class StateRepresentation:
    @staticmethod
    def sum(v1, v2):
        raise NotImplementedError()

    @staticmethod
    def size():
        raise NotImplementedError()

    @staticmethod
    def validate(v):
        return v

    @staticmethod
    def convert_from_quaternions(q):
        raise NotImplementedError()

    @staticmethod
    def convert_to_quaternions(q):
        raise NotImplementedError()
