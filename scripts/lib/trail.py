import numpy as np
import manim as m

class RayObject(m.VMobject):
	"""A named class of VMobject for debugging purposes"""


def make_trail(dot: m.Dot, starting_point: tuple[float, float, float] | np.ndarray, **kwargs) -> RayObject:
	"""Make a trail tracking the given dot beginning at the starting point"""
	trail = RayObject()
	trail.add_updater(lambda x: x.become(
		m.Line(starting_point, dot, **kwargs)
	))
	return trail

## TODO Change the above so that the visual alpha of the line varies depending on distance from
## the current location -- i.e., a trajectory leaves a faded "trail".