from typing import Callable
import numpy as np
import manim as m

"""An isotopy is a function I: (x, y, z, t, dt) -> (x', y', z', t + dt) for all
	-t < dt < 1 - t. It restricts to a homotopy H(x, y, z, t) = I(x, y, z, 0, t)."""

SIN_TOLERANCE = 1e-4
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

class Wavefront:
	"""A circule wavefront bound by a fixed curve."""
	# TODO Right now it's hard-coded to be a translate of the parabola y = ax^2.
	def __init__(self, center: np.ndarray, a: float):
		self.center = center
		self.a = a
	
	def get_angle_bounds(self, radius: float):
		"""Returns the maximum and minimum angle within the envelope at the given radius.
		The minimum angle is in the range [-m.TAU / 4, m.TAU / 4]
		The maximum angle is in the range [m.TAU / 4, 3 * m.TAU / 4]
		This is the key function defining the envelope."""
		if radius <= 0.25 / self.a:
			return - m.TAU / 4, 3 * m.TAU / 4
		else:
			x = np.sqrt(4 * radius * self.a - 1) / (2 * self.a)
			y = self.a * x ** 2- 1 / (4 * self.a)
			# Arctan takes values in [-m.TAU / 4, m.TAU / 4]
			theta = np.arctan(y/x)
			return theta, m.TAU/2 - theta
	
	def make_arc(self, radius: float) -> m.Arc:
		"""Produces an Arc object."""
		min_angle, max_angle = self.get_angle_bounds(radius)
		return m.Arc(
			arc_center=self.center,
			radius=radius,
			start_angle = min_angle,
			angle = max_angle - min_angle
			)
	
	def isotopy(self, arc: m.Arc, old_radius: float, new_radius: float, **kwargs) -> Isotopy:
		"""Produces the Isotopy mapping the arc to one with the second radius."""
		def istpy(x: float, y: float, z: float, t: float, dt: float):
			"""Advances the point (x, y, z, t) by dt (whether forward or backward)."""
			if dt == 0:
				return x, y, z, t
			else:
				# Convert to polar coordinates (r, alpha) centered on the wavefront
				vec = np.array([x, y, 0]) - self.center
				r = np.linalg.norm(vec)

				# This gives a value in [-m.TAU / 4, m.TAU / 4].
				theta_y = np.arcsin(vec[1] / r)

				# We must reflect over the y-axis if the original vector had
				# negative x-coordinate.
				if vec[0] < 0:
					theta = m.TAU / 2 - theta_y
				else:
					theta = theta_y

				min_angle, max_angle = self.get_angle_bounds(r)
				alpha = (theta - min_angle) / (max_angle - min_angle)

				# Get the corresponding point on the larger arc
				target_radius = old_radius * (1 - (t + dt)) + new_radius * (t + dt)
				new_min_angle, new_max_angle = self.get_angle_bounds(target_radius)
				new_theta = alpha * new_max_angle + (1 - alpha) * new_min_angle

				# Convert back to Cartesian coordinates
				return (
					target_radius * np.cos(new_theta) + self.center[0],
					target_radius * np.sin(new_theta) + self.center[1],
					z,
					t + dt
				)
			
		return Isotopy(isotopy=istpy, mobject=arc, rate_func = m.linear, **kwargs)
	
	# def homotopy(self, arc: m.Arc, new_radius: float) -> m.Homotopy:
	# 	"""Produces the Homotopy mapping the arc to one with the second radius."""
	# 	def htpy(x: float, y: float, z: float, t: float):
	# 		if t == 0:
	# 			return x, y, z
	# 		else:
	# 			# Convert to polar coordinates (r, alpha) centered on the wavefront
	# 			r = np.linalg.norm(np.array([x, y, 0]) - self.center)
	# 			theta = np.arccos(x / r)
	# 			alpha = (theta - arc.start_angle) / arc.angle

	# 			# Get the corresponding point on the larger arc
	# 			target_radius = arc.radius * (1-t) + new_radius * t
	# 			new_min_angle, new_max_angle = self.get_angle_bounds(target_radius)
	# 			new_theta = alpha * new_max_angle + (1 - alpha) * new_min_angle

	# 			# Convert back to Cartesian coordinates
	# 			return target_radius * np.cos(new_theta) + self.center[0], target_radius * np.sin(new_theta) + self.center[1], z

	# 	return m.Homotopy(homotopy=htpy, mobject=arc, rate_func = m.linear, run_time=1.0)