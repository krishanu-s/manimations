### Functions

from __future__ import annotations
from typing import List, Optional
import numpy as np
from ray import (is_close_to, Point, Vector, Hyperplane, Ray)

ROOT_TOLERANCE = 1e-4
MAX_ROOT = 2 ** 32

class PolyFunction:
	# A polynomial function defined by its coefficients, starting from the constant term
    def __init__(self, coefficients: List[float]):
        self.coefficients = coefficients

    def evaluate(self, x: float) -> float:
        return sum(coeff * x**i for i, coeff in enumerate(self.coefficients))

    def __add__(self, other: PolyFunction) -> PolyFunction:
        max_degree = max(len(self.coefficients), len(other.coefficients))
        coefficients = [0.0] * max_degree
        for i, coeff in enumerate(self.coefficients):
            coefficients[i] += coeff
        for i, coeff in enumerate(other.coefficients):
            coefficients[i] += coeff
        return PolyFunction(coefficients)

    def __mul__(self, other: PolyFunction) -> PolyFunction:
        max_degree = len(self.coefficients) + len(other.coefficients) - 1
        coefficients = [0.0] * max_degree
        for i, coeff1 in enumerate(self.coefficients):
            for j, coeff2 in enumerate(other.coefficients):
                coefficients[i + j] += coeff1 * coeff2
        return PolyFunction(coefficients)

    def scale(self, scalar: float) -> PolyFunction:
        return PolyFunction([coeff * scalar for coeff in self.coefficients])

    def roots(self) -> List[float]:
        """Returns the roots of the equation. For numerical stability, omits any roots greater than MAX_ROOT."""
        coefficients = self.coefficients
        if len(coefficients) == 2:
            root = -coefficients[0] / coefficients[1]
            return [root]
        elif len(coefficients) == 3:
            c, b, a = coefficients
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                return []
            elif discriminant == 0:
                root = -b / (2*a)
                return [root]
            else:
                sqrt_discriminant = np.sqrt(discriminant)
                return [(-b + sqrt_discriminant) / (2*a), (-b - sqrt_discriminant) / (2*a)]
        else:
            raise ValueError("Polynomial degree must be 2 or 3")

class Conic:
	"""A conic section in the form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0"""
	a: float
	b: float
	c: float
	d: float
	e: float
	f: float

	def __init__(self, a: float, b: float, c: float, d: float, e: float, f: float):
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.e = e
		self.f = f

	def evaluate(self, point: Point) -> float:
		x, y = point.x, point.y
		return self.a * x**2 + self.b * x * y + self.c * y**2 + self.d * x + self.e * y + self.f

	def restrict(self, ray: Ray) -> PolyFunction:
		# Constructs to a one-variable quadratic function on the given ray
		# Points on the ray are parametrized by a new variable t. The quadratic function,
		# restricted to the ray, has the form At^2 + Bt + C = 0
		x0 = ray.start.x
		y0 = ray.start.y
		vx = ray.direction.x
		vy = ray.direction.y

		ylin = PolyFunction([y0, vy])
		xlin = PolyFunction([x0, vx])

		# Start with constant term
		polynomial = PolyFunction([self.f])

		# Add linear terms
		polynomial += ylin.scale(self.e)
		polynomial += xlin.scale(self.d)

		# Add quadratic terms
		polynomial += (ylin * ylin).scale(self.c)
		polynomial += (xlin * ylin).scale(self.b)
		polynomial += (xlin * xlin).scale(self.a)

		return polynomial

	def intersection(self, ray: Ray) -> float | None:
		"""Calculates the first t-value where the ray intersects the conic. If there is no intersection,
		returns np.inf. If the nearest intersection is too small for numerical precision purposes,
		returns None"""
		# Restrict the polynomial to the ray
		polynomial = self.restrict(ray)

		# Solve the quadratic equation
		roots = polynomial.roots()

		# Find the smallest positive root within tolerance
		min_positive_root = np.inf
		min_tolerated_root = np.inf
		for root in roots:
			if (root > ROOT_TOLERANCE) and (root < min_tolerated_root):
				min_tolerated_root = root
			if root > 0 and root < min_positive_root:
				min_positive_root = root
		match (min_tolerated_root, min_positive_root):
			case (np.inf, np.inf):
				return np.inf
			case (np.inf, min_positive_root):
				return None
			case (min_tolerated_root, _):
				return min_tolerated_root

	def tangent(self, point: Point) -> Hyperplane:
		"""Produces the tangent hyperplane at the given point on the curve."""
		# 0 = dP(x, y) = (2Ax + By + D)dx + (Bx + 2Cy + E)dy
		normal = Vector(
			2.0 * self.a * point.x + self.b * point.y * self.d,
			self.b * point.x + 2.0 * self.c * point.y + self.e,
		)
		return Hyperplane(normal, point)

	# TODO Deprecate.
	def bounce(self, ray: Ray) -> Ray | None:
		"""Follows the given ray until it intersects the curve again, and then reflects through the tangent
		hyperplane at the intersection point to produce a new ray."""
		t = self.intersection(ray)
		if t is np.inf:
			return None

		intersection = ray.point_at(t)
		tangent = self.tangent(intersection)
		reflected_vector = tangent.reflect(Ray.direction)
		return Ray(intersection, reflected_vector)

	def make_trajectory(self, start: Point, direction: Vector, total_dist: float):
		"""Given a starting point and direction, produces a sequence of points such that the connecting line
		segments are the trajectory of the point as it bounces off the fixed conic."""
		remaining_dist = total_dist
		points = [start.copy()]
		ray = Ray(start, direction)
		while True:
			next_t = self.intersection(ray)

			# The next point is too small for numerical precision -- end the process here.
			if next_t is None:
				break

			# If there is no intersection, go out the remaining distance in that direction.
			if next_t is np.inf:
				points.append(ray.point_at_dist(remaining_dist))
				break

			vec = ray.direction * next_t
			dist = vec.magnitude()

			# If there is no intersection within the remaining distance, go out the remaining distance in that direction.
			if dist > remaining_dist:
				points.append(ray.point_at_dist(remaining_dist))
				break

			# Otherwise, go to the next point and update the ray
			next_point = ray.point_at(next_t)
			tangent = self.tangent(next_point)
			reflected_vector = tangent.reflect(ray.direction)
			ray = Ray(ray.point_at(next_t), reflected_vector)

			points.append(next_point)
			remaining_dist -= dist


		return points
