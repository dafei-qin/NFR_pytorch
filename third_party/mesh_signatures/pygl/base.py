import numpy as np
from OpenGL.GL import *
import logging

class GLObject(object):
    def __init__(self, constructor=None, destructor = None):
        self._id = constructor(1) if constructor is not None else 0

        self._destructor = destructor
        ctx = Context.current()
        assert ctx is not None, "No current context"
        ctx._register_created_object(self)

    def free(self):
        if self._id != 0 and self._destructor is not None:
            logging.debug("GLObject.free")
            self._destructor(1, np.array(self._id))
            self._id = 0

    @property
    def id(self):
        return self._id

class Context(object):
    context_stack = []

    @staticmethod
    def current():
        return Context.context_stack[-1] if len(Context.context_stack) > 0 else None 

    def __init__(self):
        self._created_objects = []
        self.fbo_stack = []
        self._dummy_vao = 0

    def __del__(self):
        logging.debug("Context.__del__")
        with self as ctx:
            for obj in ctx._created_objects:
                logging.debug(f"Context.__del__: free {obj.__class__.__name__}:{obj.id}")
                obj.free()
            self._created_objects = []
        # remove every other reference in context stack
        Context.context_stack = [ctx for ctx in Context.context_stack if ctx != self]
        if self._dummy_vao != 0:
            glDeleteVertexArrays(1, np.array(self._dummy_vao))

    def set_active(self):
        raise NotImplementedError("Do not use context classes that do not implement set_active!")
    
    def dismiss(self):
        assert Context.context_stack[-1] == self, "Context stack was corrupted"
        Context.context_stack = Context.context_stack[:-1]
        if len(Context.context_stack) > 0:
            Context.context_stack.set_active(Context.context_stack[-1])

    def __enter__(self):
        logging.debug("Context.__enter__")
        self.set_active()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        logging.debug("Context.__exit__")
        self.dismiss()

    def _register_created_object(self, obj):
        assert isinstance(obj, GLObject), "Tried to register instance of a non GLObject"
        self._created_objects.append(obj)

    @property 
    def dummy_vao(self):
        if self._dummy_vao == 0:
            self._dummy_vao = glGenVertexArrays(1)
        return self._dummy_vao