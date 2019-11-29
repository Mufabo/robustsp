
Module defining global singleton classes.

This module raises a RuntimeError if an attempt to reload it is made. In that
way the identities of the classes defined here are fixed and will remain so
even if numpy itself is reloaded. In particular, a function like the following
will still work correctly after numpy is reloaded::

    def foo(arg=np._NoValue):
        if arg is np._NoValue:
            ...

That was not the case when the singleton classes were defined in the numpy
``__init__.py`` file. See gh-7844 for a discussion of the reload problem that
motivated this module.


numpy._globals
**************

.. toctree::
   :maxdepth: 5

    
   