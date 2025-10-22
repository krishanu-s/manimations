### Conic sections in the (projective) plane

from __future__ import annotations
from enum import Enum
from typing import Callable
import numpy as np
from .point import Point2D, ProjectivePoint
from .envelope import BoundsFn
from .ray import Point3D, Vector3D, Hyperplane, Ray
from .polyfunction import PolyFunction

ROOT_TOLERANCE = 1e-4
MAX_ROOT = 2 ** 32
COEFF_TOLERANCE = 1e-5
ANGLE_TOLERANCE = 1e-5

def isclose(a: float, b: float):
    """Tolerances for computation"""
    return abs(a - b) < COEFF_TOLERANCE

### animation aid

"""An isotopy is a function I: (x, y, z, t, dt) -> (x', y', z', t + dt) for all
	-t < dt < 1 - t. It restricts to a homotopy H(x, y, z, t) = I(x, y, z, 0, t)."""
IsotopyFn = Callable[[float, float, float, float, float], tuple[float, float, float]]


### Types for specifying a conic section

class PolarConicEquation:
    """An equation for r in terms of θ of the form r = C / (1 + E * cos(θ - θ_0)).
    Invariant under sending E -> -E, and θ_0 -> θ_0 + π, and also
    invariant under sending C -> -C and θ_0 -> θ_0 + π.
    So we assume E >= 0 and C > 0."""
    focus: Point2D
    e: float
    c: float
    theta_0: float
    other_focus: ProjectivePoint

    def __init__(
        self,
        focus: Point2D,
        e: float,
        c: float,
        theta_0: float
    ):
        assert e >= 0, "Eccentricity must be nonnegative."
        assert c > 0, "Scale must be positive"
        assert theta_0 >= 0 and theta_0 < (2 * np.pi), "Angle must be in [0, 2 * π)"
        
        self.focus = focus
        self.e = e
        self.c = c
        self.theta_0 = theta_0
        self.other_focus = self._calculate_other_focus()

    def __eq__(self, other: PolarConicEquation):
        return (self.focus == other.focus) and isclose(self.e, other.e) and isclose(self.c, other.c) and isclose(self.theta_0 - other.theta_0 % (2 * np.pi), 0)
    
    def __repr__(self):
        return f"Focus={self.focus}, Eccentricity={self.e}, Scale={self.c}, Angle={self.theta_0}"

    def _calculate_other_focus(self) -> ProjectivePoint:
        """Calculates the position of the other focus from the given data, in absolute coordinates"""
        if self.e == 1:
            return ProjectivePoint(np.cos(self.theta_0), np.sin(self.theta_0), 0)
        elif self.e == 0:
            return self.focus.to_projective()
        else:
            # Signed distance between foci
            dist = self.c * (1/(self.e + 1) + 1/(self.e - 1))
            return ProjectivePoint(
                x=self.focus.x + dist * np.cos(self.theta_0),
                y=self.focus.y + dist * np.sin(self.theta_0),
                z=1
            )

    def bounds(self) -> BoundsFn:
        """Returns a function which takes radii r as input, and outputs the minimum and
        maximum values of theta on the arc at distance r from the focus and in the same
        plane region as the focus."""
        def f(radius: float):
            if radius < self.c / (1 + self.e):
                # angle = 0
                return (self.theta_0, self.theta_0 + 2 * np.pi)
            elif radius > self.c / (1 - self.e):
                # angle = π
                return (self.theta_0 + np.pi - ANGLE_TOLERANCE, self.theta_0 + np.pi + ANGLE_TOLERANCE)
            else:
                # Solve the defining equation for theta
                cos_value = (-1 + self.c / radius) / self.e
                assert abs(cos_value) <= 1
                angle = np.arccos(cos_value)
                # Setting θ = θ_0 yields the smallest possible radius, so the arc must go from
                # θ_0 + angle to θ_0 + 2π - angle
                return (self.theta_0 + angle, self.theta_0 + 2 * np.pi - angle)

        return f

    def to_cartesian(self) -> CartesianConicEquation:
        """Generates the Cartesian form."""
        if self.e == 0:
            cart_eq = CartesianConicEquation(
                c_xx = 1 / (self.c**2),
                c_xy = 0.,
                c_yy = 1 / (self.c**2),
                c_x=0.,
                c_y=0.,
                c_0=-1.
            )
        else:
            cart_eq = CartesianConicEquation(
                c_xx = (1 / self.e ** 2) - 1,
                c_xy = 0.,
                c_yy = (1 / self.e ** 2),
                c_x = 2 * self.c / self.e,
                c_y = 0,
                c_0 = -(self.c/self.e) ** 2
            )
            cart_eq.rotate(self.theta_0)
        cart_eq.translate_x(self.focus.x)
        cart_eq.translate_y(self.focus.y)
        return cart_eq
    
    def translate_x(self, a: float):
        """Substitutes x -> (x - a), translating the graph a units rightward."""
        self.focus.translate_x(a)
        self.other_focus.translate_x(a)
    
    def translate_y(self, a: float):
        """Substitutes y -> (y - c), translating the graph c units upward."""
        self.focus.translate_y(a)
        self.other_focus.translate_y(a)
    
    def rotate(self, theta: float):
        """Rotates the graph counterclockwise around the focus by θ by applying the substitution
        x ->  x * cos(θ) + y * sin(θ)
        y -> -x * sin(θ) + y * cos(θ)
        """
        self.theta_0 -= theta
        self.other_focus.translate_x(-self.focus.x)
        self.other_focus.translate_y(-self.focus.y)
        self.other_focus.rotate(theta)
        self.other_focus.translate_x(self.focus.x)
        self.other_focus.translate_y(self.focus.y)
    
    @classmethod
    def std_parabola(cls, m: float) -> PolarConicEquation:
        """Produces the polar form of the parabola my^2 = x, which has
        focus at (1/4m, 0) and vertex at (0, 0)."""
        assert m != 0
        return PolarConicEquation(
            focus=Point2D(x=0.25 / m, y=0),
            e=1.0,
            c=-0.5 / m,  # If c = 2, focal length would be 1
            theta_0=0
            )
    
    @classmethod
    def std_ellipse(cls, a: float, b: float) -> PolarConicEquation:
        """Produces the polar form of the ellipse (x/a)^2 + (y/b)^2 = 1, where a >= b."""
        assert abs(a) >= abs(b), "Major axis must lie along the x-axis."
        if a == b:
            return PolarConicEquation(
                focus=Point2D(x=0, y=0),
                e=0,
                c=a**2,
                theta_0=0
            )
        length = np.sqrt(a**2 - b**2)
        return PolarConicEquation(
            focus=Point2D(x=length, y=0),
            e=length/a,
            c=(b**2)/a,
            theta_0=0
            )
    
    @classmethod
    def std_hyperbola(cls, a: float, b: float) -> PolarConicEquation:
        """Produces the polar form of the hyperbola (x/a)^2 - (y/b)^2 = 1."""
        length = np.sqrt(a**2 + b**2)
        return PolarConicEquation(
            focus=Point2D(x=-length, y=0),
            e=length/a,
            c=(b**2)/a,
            theta_0=0
            )

class CartesianConicEquation:
    """An equation relating x and y in the form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0,
    where at least one of A, B, C, D, E is nonzero.
    Invariant under scaling all parameters, so we assume that self.c_xx + self.c_yy >= 0."""
    c_xx: float
    c_xy: float
    c_yy: float
    c_x: float
    c_y: float
    c_0: float

    def __init__(
        self,
        c_xx: float = 0.0,
        c_xy: float = 0.0,
        c_yy: float = 0.0,
        c_x: float = 0.0,
        c_y: float = 0.0,
        c_0: float = 0.0
    ):
        assert any(coeff != 0 for coeff in (c_xx, c_xy, c_yy, c_x, c_y)), "Must have at least one non-zero non-constant coefficient"
        scale = -1.0 if c_xx + c_yy < 0 else 1.0
        self.c_xx = c_xx * scale
        self.c_xy = c_xy * scale
        self.c_yy = c_yy * scale
        self.c_x = c_x * scale
        self.c_y = c_y * scale
        self.c_0 = c_0 * scale

    def __repr__(self):
        return f"({self.c_xx:.4f}, {self.c_xy:.4f}, {self.c_yy:.4f}, {self.c_x:.4f}, {self.c_y:.4f}, {self.c_0:.4f})"

    def __eq__(self, other: CartesianConicEquation):
        if self.c_xx != 0:
            scale = other.c_xx / self.c_xx
        elif self.c_xy != 0:
            scale = other.c_xy / self.c_xy
        elif self.c_yy != 0:
            scale = other.c_yy / self.c_yy
        elif self.c_x != 0:
            scale = other.c_x / self.c_x
        elif self.c_y != 0:
            scale = other.c_y / self.c_y
        else:
            return False
        return all(
            isclose(scale * getattr(self, name), getattr(other, name))
            for name in ('c_xx', 'c_xy', 'c_yy', 'c_x', 'c_y', 'c_0')
        )


    def clone(self) -> CartesianConicEquation:
        return CartesianConicEquation(self.c_xx, self.c_xy, self.c_yy, self.c_x, self.c_y, self.c_0)

    def to_polar(self) -> PolarConicEquation:
        """Generates the polar form of the equation."""
        cart_eq = self.clone()
        
        # First apply a rotation by θ to eliminate the c_xy term, using
        # tan(2θ) = c_xy / (c_xx - c_yy)
        if cart_eq.c_xx == cart_eq.c_yy:
            theta = np.pi/4
        else:
            theta = np.arctan(cart_eq.c_xy / (cart_eq.c_xx - cart_eq.c_yy)) / 2
        cart_eq.rotate(theta)

        # At this point, either c_xx is nonzero or c_yy is nonzero.
        # Rotate by π/2 if necessary to ensure |c_yy| >= |c_xx|.
        if abs(cart_eq.c_yy) < abs(cart_eq.c_xx):
            cart_eq.rotate(np.pi/2)
            theta += np.pi/2
        
        # Translate vertically to eliminate the c_y term
        ay = cart_eq.c_y / (2 * cart_eq.c_yy)
        cart_eq.translate_y(ay)

        # Case 1: If c_xx == 0, then the result is a parabola Cy^2 + Dx + F = 0.
        if cart_eq.c_xx == 0:
            # Translate x to eliminate the constant term F, and thus vertex at (0, 0)
            ax = cart_eq.c_0 / cart_eq.c_x
            cart_eq.translate_x(ax)
            
            # Now the equation is (-C/D)y^2 = x, so retrieve the equation
            polar_eq = PolarConicEquation.std_parabola(-cart_eq.c_yy / cart_eq.c_x)

        # Case 2: If c_xx != 0, then the result is an ellipse or hyperbola
        else:
            # Translate x to eliminate the c_x term and thus center at (0, 0).
            ax = cart_eq.c_x / (2 * cart_eq.c_x)
            cart_eq.translate_x(ax)

            # Now the equation is Ax^2 + Cy^2 + F = 0, so retrieve the equation
            if cart_eq.c_xx * cart_eq.c_yy > 0:
                polar_eq = PolarConicEquation.std_ellipse(
                    a=np.sqrt(-cart_eq.c_0 / cart_eq.c_xx),
                    b=np.sqrt(-cart_eq.c_0 / cart_eq.c_yy)
                    )
            elif cart_eq.c_xx * cart_eq.c_yy < 0:
                polar_eq = PolarConicEquation.std_hyperbola(
                    a=np.sqrt(-cart_eq.c_0 / cart_eq.c_xx),
                    b=-np.sqrt(-cart_eq.c_0 / cart_eq.c_yy)
                    )
        
        # Perform the reverse operations on the polar equation
        polar_eq.translate_x(-ax)
        polar_eq.translate_y(-ay)
        polar_eq.rotate(-theta)
        return polar_eq
    
    def translate_x(self, a: float):
        """Substitutes x -> (x - a), translating the graph a units rightward."""
        c_0 = self.c_0 - a * self.c_x + (a**2) * self.c_xx
        c_x = self.c_x - 2 * a * self.c_xx
        c_y = self.c_y - a * self.c_xy
        self.c_0 = c_0
        self.c_x = c_x
        self.c_y = c_y
    
    def translate_y(self, a: float):
        """Substitutes y -> (y - c), translating the graph c units upward."""
        c_0 = self.c_0 - a * self.c_y + (a**2) * self.c_yy
        c_y = self.c_y - 2 * a * self.c_yy
        c_x = self.c_x - a * self.c_xy
        self.c_0 = c_0
        self.c_y = c_y
        self.c_x = c_x
    
    def rotate(self, theta: float):
        """Rotates the graph counterclockwise by θ by applying the substitution
        x ->  x * cos(θ) + y * sin(θ)
        y -> -x * sin(θ) + y * cos(θ)
        """
        c, s = np.cos(theta), np.sin(theta)

        c_xx = self.c_xx * (c**2) + self.c_yy * (s**2) - self.c_xy * c * s
        c_yy = self.c_yy * (c**2) + self.c_xx * (s**2) + self.c_xy * c * s
        c_xy = (self.c_xx - self.c_yy) * np.sin(2 * theta) + self.c_xy * np.cos(2 * theta)
        self.c_xx = c_xx
        self.c_yy = c_yy
        self.c_xy = c_xy

        c_x = self.c_x * c - self.c_y * s
        c_y = self.c_x * s + self.c_y * c
        self.c_x = c_x
        self.c_y = c_y
    
    @classmethod
    def std_parabola(cls, m: float) -> CartesianConicEquation:
        """Produces the polar form of the parabola my^2 = x, which has
        focus at (1/4m, 0) and vertex at (0, 0)."""
        assert m != 0
        return CartesianConicEquation(0., 0., m, -1., 0., 0.)
    
    @classmethod
    def std_ellipse(cls, a: float, b: float) -> CartesianConicEquation:
        """Produces the polar form of the ellipse (x/a)^2 + (y/b)^2 = 1, where a >= b."""
        assert abs(a) >= abs(b), "Major axis must lie along the x-axis."
        return CartesianConicEquation(1/(a**2), 0., 1/(b**2), 0., 0., -1.)
    
    @classmethod
    def std_hyperbola(cls, a: float, b: float) -> CartesianConicEquation:
        """Produces the polar form of the hyperbola (x/a)^2 - (y/b)^2 = 1."""
        return CartesianConicEquation(1/(a**2), 0., -1/(b**2), 0., 0., -1.)
    
    ### For animating trajectories

    def restrict(self, ray: Ray) -> PolyFunction:
        """
        Constructs to a one-variable quadratic function on the given ray. Since points
        on the ray are parametrized by a new variable t, the quadratic function has
        the form At^2 + Bt + C = 0
        """
        x0 = ray.start.x
        y0 = ray.start.y
        vx = ray.direction.x
        vy = ray.direction.y

        ylin = PolyFunction([y0, vy])
        xlin = PolyFunction([x0, vx])

		# Start with constant term
        polynomial = PolyFunction([self.c_0])

		# Add linear terms
        polynomial += ylin.scale(self.c_y)
        polynomial += xlin.scale(self.c_x)

		# Add quadratic terms
        polynomial += (ylin * ylin).scale(self.c_yy)
        polynomial += (xlin * ylin).scale(self.c_xy)
        polynomial += (xlin * xlin).scale(self.c_xx)

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

    def tangent(self, point: Point3D) -> Hyperplane:
        """Produces the tangent hyperplane at the given point on the curve."""
		# 0 = dP(x, y) = (2Ax + By + D)dx + (Bx + 2Cy + E)dy
        normal = Vector3D(
			2.0 * self.c_xx * point.x + self.c_xy * point.y * self.c_x,
			self.c_xy * point.x + 2.0 * self.c_yy * point.y + self.c_y,
		)
        return Hyperplane(normal, point)

    def make_trajectory(self, start: Point, direction: Vector, total_dist: float):
        """
        Given a starting point and direction, produces a sequence of points such that the connecting line
        segments are the trajectory of the point as it bounces off the fixed conic.
        """
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

class ConicType(Enum):
    """Types of conic based on eccentricity."""
    Ellipse = 1
    Parabola = 2
    Hyperbola = 3

class ConicSection:
    """
    A conic section can be defined either from its Cartesian form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0,
    or from its polar-coordinates form r = C / (1/E + cos(θ - θ_0)) around one focus.

    We initialize a Conic from one of these two forms, rather than calling __init__ directly.
    """
    def __init__(
        self,
        polar_eq: PolarConicEquation,
        cart_eq: CartesianConicEquation
    ):
        self.polar_eq: PolarConicEquation = polar_eq
        self.cart_eq: CartesianConicEquation = cart_eq

        # Define the foci
        self.focus = polar_eq.focus
        self.other_focus = polar_eq.other_focus

        # Define the eccentricity
        self.eccentricity = polar_eq.e
        assert self.eccentricity > 0
    
    def _type(self) -> ConicType:
        """Returns the type of conic."""
        if self.eccentricity == 1:
            return ConicType.Parabola
        elif self.eccentricity < 1:
            return ConicType.Ellipse
        elif self.eccentricity > 1:
            return ConicType.Hyperbola

    @classmethod
    def from_cartesian(cls, cart_eq: CartesianConicEquation) -> ConicSection:
        """Initializes a conic section from a two-variable quadratic equation."""
        return ConicSection(polar_eq=cart_eq.to_polar(), cart_eq=cart_eq)

    @classmethod
    def from_polar(cls, polar_eq: PolarConicEquation) -> ConicSection:
        """Given one focus, the conic may be defined by a function of polar coordinates
        (r, θ) centered around that focus of the form
        r = C / (1/E + cos(θ - θ_0))"""
        return ConicSection(polar_eq=polar_eq, cart_eq=polar_eq.to_cartesian())

    ### For animating wavefronts


# Tests.
if __name__ == "__main__":
    # Parabola with focus at (0, 0) and focal length 1
    polar_eq = PolarConicEquation(Point2D(0, 0), 1, 2, 0)
    cart_eq = polar_eq.to_cartesian()
    assert cart_eq == CartesianConicEquation(0., 0., 1., 4., 0., -4.)
    new_polar_eq = cart_eq.to_polar()
    assert polar_eq == new_polar_eq

    # Circle
    polar_eq = PolarConicEquation.std_ellipse(1.0, 1.0)
    assert polar_eq.other_focus == ProjectivePoint(0, 0, 1)
    cart_eq = CartesianConicEquation.std_ellipse(1.0, 1.0)
    assert cart_eq == polar_eq.to_cartesian()

    # Standard ellipse
    polar_eq = PolarConicEquation.std_ellipse(5.0, 3.0)
    assert polar_eq.other_focus == ProjectivePoint(-4, 0, 1)
    cart_eq = CartesianConicEquation.std_ellipse(5.0, 3.0)
    assert cart_eq == polar_eq.to_cartesian()

    # Standard hyperbola
    polar_eq = PolarConicEquation.std_hyperbola(5.0, 3.0)
    assert polar_eq.other_focus == ProjectivePoint(np.sqrt(34), 0, 1)
    cart_eq = CartesianConicEquation.std_hyperbola(5.0, 3.0)
    assert cart_eq == polar_eq.to_cartesian()