"""
An isotopy is a function I: (x, y, z, t, dt) -> (x', y', z', t + dt) for all
-t < dt < 1 - t. It restricts to a homotopy H(x, y, z, t) = I(x, y, z, 0, t).
"""

from typing import Callable
import manim as m

IsotopyFn = Callable[[float, float, float, float, float], tuple[float, float, float]]

class Isotopy(m.Homotopy):
	
	def __init__(self, isotopy: IsotopyFn, run_time: float = 3, **kwargs):
		self.isotopy = isotopy
		# Keep a copy of initialization kwargs for re-initialization
		self.kwargs = kwargs
		def homotopy(x, y, z, t):
			x1, y1, z1, t1 = isotopy(x, y, z, 0, t)
			return x1, y1, z1
		super().__init__(homotopy=homotopy, run_time=run_time, **kwargs)