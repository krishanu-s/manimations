from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DISTANCE_TOLERANCE = 1e-5

def isclose(a: float, b: float):
    """Tolerances for computation"""
    return abs(a - b) < DISTANCE_TOLERANCE

@dataclass
class Point2D:
    """A 2-tuple representing a point in R^2."""
    x: float
    y: float
    def translate_x(self, a: float):
        self.x += a
    def translate_y(self, a: float):
        self.y += a
    def rotate(self, theta: float):
        c, s = np.cos(theta), np.sin(theta)
        self.x, self.y = c * self.x - s * self.y, s * self.x + c * self.y
    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(self.x + other.x, self.y + other.y)
    def __mul__(self, scalar: float) -> Point2D:
        return Point2D(self.x * scalar, self.y * scalar)
    def __eq__(self, other: Point2D):
        return isclose(self.x, other.x) and isclose(self.y, other.y)
    def __repr__(self):
        return f"({self.x:.3f}, {self.y:.3f})"
    def to_projective(self) -> ProjectivePoint:
        return ProjectivePoint(self.x, self.y, 1)
    def to_triple(self) -> tuple[float, float, float]:
        return (self.x, self.y, 0)

@dataclass
class ProjectivePoint:
    """A 3-tuple [x:y:z] representing a point (x/z, y/z) in the projective plane."""
    x: float
    y: float
    z: float
    def __repr__(self):
        return f"({self.x:.3f}: {self.y:.3f}: {self.z:.3f})"
    def __eq__(self, other: ProjectivePoint):
        if self.x != 0:
            scale = other.x / self.x
        elif self.y != 0:
            scale = other.y / self.y
        elif self.z != 0:
            scale = other.z / self.z
        return all(
            isclose(scale * getattr(self, name), getattr(other, name))
            for name in ('x', 'y', 'z')
            )
    def translate_x(self, a: float):
        if self.z != 0:
            self.x += a * self.z
    def translate_y(self, a: float):
        if self.z != 0:
            self.y += a * self.z
    def rotate(self, theta: float):
        self.x, self.y = np.cos(theta) * self.x - np.sin(theta) * self.y, np.sin(theta) * self.x + np.cos(theta) * self.y
    def to_cartesian(self) -> Point2D:
        """Converts to Cartesian coordinates."""
        assert self.z != 0, "Cannot convert the point at infinity to Cartesian coordinates."
        return Point2D(self.x / self.z, self.y / self.z)
    

if __name__ == "__main__":
	# TODO Add tests.
	pass