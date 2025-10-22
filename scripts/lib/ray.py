### 2D and 3D Euclidean geometry

from __future__ import annotations
import numpy as np

# Tolerances for floating point comparisons
TOLERANCE = 1e-6

def is_close_to(x0: float, x1: float) -> bool:
    return abs(x0 - x1) < TOLERANCE

class Point3D:
	"""A point in 3D space."""
	x: float
	y: float
	z: float

	def __init__(self, x: float, y: float, z: float = 0.0):
		self.x = x
		self.y = y
		self.z = z

	def __add__(self, other: Point3D) -> Point3D:
		return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

	def __sub__(self, other: Point3D) -> Point3D:
		return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

	def __mul__(self, scalar: float) -> Point3D:
		return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

	def __repr__(self) -> str:
		return f"Point({self.x}, {self.y}, {self.z})"

	def coords(self) -> tuple[float, float, float]:
		return self.x, self.y, self.z

	def vector_to(self, other: Point3D) -> Vector3D:
		return Vector3D(other.x - self.x, other.y - self.y, other.z - self.z)

	def distance_to(self, other: Point3D) -> float:
		return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

	def copy(self) -> Point3D:
		return Point3D(self.x, self.y, self.z)

	def to_array(self) -> np.ndarray:
		return np.array([self.x, self.y, self.z])
	
	@classmethod
	def from_array(cls, arr: np.ndarray) -> Point3D:
		return Point3D(arr[0], arr[1], arr[2])

ORIGIN = Point3D(0, 0, 0)

class Vector3D:
	"""A vector in 3D space."""
	x: float
	y: float
	z: float

	def __init__(self, x: float, y: float, z: float = 0.0):
		self.x = x
		self.y = y
		self.z = z

	def __add__(self, other: Vector3D) -> Vector3D:
		return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

	def __sub__(self, other: Vector3D) -> Vector3D:
		return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

	def __mul__(self, scalar: float) -> Vector3D:
		return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

	def __repr__(self) -> str:
		return f"Vector({self.x}, {self.y}, {self.z})"

	def dot(self, other: Vector3D) -> float:
		return self.x * other.x + self.y * other.y + self.z * other.z

	def magnitude(self) -> float:
		return np.sqrt(self.x**2 + self.y**2 + self.z**2)

	def normalize(self) -> Vector3D:
		mag = self.magnitude()
		return Vector3D(self.x / mag, self.y / mag, self.z / mag)

	def project(self, other: Vector3D) -> Vector3D:
		scalar = self.dot(other) / other.dot(other)
		return other * scalar

	def is_close_to(self, other: Vector3D) -> bool:
		return abs(self.x - other.x) < TOLERANCE and abs(self.y - other.y) < TOLERANCE and abs(self.z - other.z) < TOLERANCE

	def is_same_direction(self, other: Vector3D) -> bool:
		val = self.dot(other) / (self.magnitude() * other.magnitude())
		return is_close_to(val, 1.0)

	def is_collinear(self, other: Vector3D) -> bool:
		val = self.dot(other) / (self.magnitude() * other.magnitude())
		return is_close_to(val, 1.0) | is_close_to(val, -1.0)

	def copy(self) -> Vector3D:
		return Vector3D(self.x, self.y, self.z)

class Ray:
	"""A ray, defined by a starting point p and a direction vector v. Points on the ray are parametrized by
	nonnegative real numbers t according to the equation P(t) = p + tv"""
	start: Point3D
	direction: Vector3D
	def __init__(self, start: Point3D, direction: Vector3D):
		self.start = start
		self.direction = direction
	def __repr__(self):
		return f"Ray(start={self.start}, direction={self.direction})"
		def copy(self) -> Ray:
			return Ray(self.start.copy(), self.direction.copy())
	def point_at(self, t: float) -> Point3D:
		"""Returns the point on the ray at time t."""
		return Point3D(
			x = self.start.x + self.direction.x * t,
			y = self.start.y + self.direction.y * t,
			z = self.start.z + self.direction.z * t,
		)
	def point_at_dist(self, distance: float) -> Point3D:
		"""Returns the point on the ray a given distance from the start point."""
		return self.point_at(distance / self.direction.magnitude())
	def contains(self, point: Point3D) -> bool:
		"""Checks whether the given point lies on the ray"""
		point_vector = self.start.vector_to(point)
		return point_vector.is_same_direction(self.direction)

class Hyperplane:
	"""A hyperplane defined by a point on the hyperplane and a normal vector."""
	normal: Vector3D
	point: Point3D

	def __init__(self, normal: Vector3D, point: Point3D):
		self.normal = normal
		self.point = point

	def reflect(self, vector: Vector3D) -> Vector3D:
		"""Reflect the chosen vector over the hyperplane"""
		return vector - (vector.project(self.normal) * 2.0)

	def intersect(self, ray: Ray) -> float:
		"""Calculates the first t-value where the ray intersects the hyperplane.
		Returns np.inf if there is no intersection."""

		denominator = self.normal.dot(ray.direction)
		if abs(denominator) < TOLERANCE:
			return np.inf  # Parallel or coincident trajectories

		numerator = self.normal.dot(ray.start.vector_to(self.point))
		t = numerator / denominator

		if t < 0:
			return np.inf  # Intersection point is behind the ray's start point

		return t


if __name__ == "__main__":
	# TODO Add tests.
	pass