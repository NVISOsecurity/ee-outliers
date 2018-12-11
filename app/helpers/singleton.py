def singleton(cls):
    """ decorator for a class to make a singleton out of it """
    class_instance = {}

    def get_instance(*args, **kwargs):
        """ creating or just return the one and only class instance.
            The singleton depends on the parameters used in __init__ """
        key = (cls, args, str(kwargs))
        if key not in class_instance:
            class_instance[key] = cls(*args, **kwargs)
        return class_instance[key]

    return get_instance
