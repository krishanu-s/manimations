### Polynomial functions

from __future__ import annotations
from typing import List
import numpy as np

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
