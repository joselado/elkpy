"""Task-family mixins for :class:`elkpy.calculation.Calculation`.

One module per Elk task family (see ``vendor/elk/src/elk.f90``'s dispatch).
Each defines a mixin class of ``get_*`` methods that run through
``Calculation._run_resumed`` (or, for the tasks that drive their own
sequences of ground states, through that family's own runner);
``Calculation`` inherits them all.

``ALL_MIXINS`` is the single place the composition is written down --
``calculation.py`` unpacks it into ``class Calculation(*ALL_MIXINS)``, so
adding a family means adding one module and one entry here. No mixin
imports ``..calculation``, so there is no import cycle; none defines
``__init__`` or state of its own, so the order below is documentation
rather than semantics.

The one shared piece deliberately NOT in a mixin is ``_ndmag()`` (with its
``_block_floats``/``_block_flag`` helpers): several families need Elk's
``ndmag`` decision from src/init0.f90, and two independent transcriptions
of one Fortran rule drift apart, so it lives on ``Calculation`` itself.
"""

from .groundstate import GroundStateResponseTasks
from .magnetism_manybody import MagnetismManyBodyTasks
from .optics import OpticsTasks
from .params import ParametersMixin
from .phonons import PhononTasks
from .spectra import SpectraTasks

#: Every task-family mixin, in the order Calculation inherits them.
ALL_MIXINS = (
    ParametersMixin,             # the elk.in input surface (no task of its own)
    GroundStateResponseTasks,    # 5, 68, 110, 115, 190, 195/196, 300, 380, 390,
                                 # 400, 420/421, 430, 440, 500
    SpectraTasks,                # 10, 14-16, 21-24, 31/32, 41/42, 51/52, 61-65,
                                 # 71-93, 100-105, 140-153, 162, 170-173,
                                 # 341-343, 371-373, 471
    OpticsTasks,                 # 125, 130, 135, 180, 185-187, 285, 320,
                                 # 330/331, 450-456, 460-463, 480/481
    PhononTasks,                 # 200-202, 208/209, 230, 240/241, 245, 250,
                                 # 260, 270, 280, 478
    MagnetismManyBodyTasks,      # 28/29, 160, 350-352, 550, 600-640, 700-773
)

__all__ = [
    "ALL_MIXINS",
    "GroundStateResponseTasks",
    "MagnetismManyBodyTasks",
    "OpticsTasks",
    "ParametersMixin",
    "PhononTasks",
    "SpectraTasks",
]
