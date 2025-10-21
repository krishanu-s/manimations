
from __future__ import annotations
from enum import Enum
from typing import Callable
from dataclasses import dataclass
import numpy as np
from ray import Point, Vector, Hyperplane, Ray
from polyfunction import PolyFunction

ROOT_TOLERANCE = 1e-4
MAX_ROOT = 2 ** 32
def isclose(a: float, b: float):
    """Tolerances for computation"""
    return abs(a - b) < 1e-5

@dataclass
class Point2D:
    """A 2-tuple representing a point in R^2."""
    x: float
    y: float
    def translate_x(self, a: float):
        self.x += a
    def translate_y(self, a: float):
        self.y += a
    def __eq__(self, other: Point2D):
        return isclose(self.x, other.x) and isclose(self.y, other.y)
    def __repr__(self):
        return f"({self.x:.3f}, {self.y:.3f})"

@dataclass
class ProjectivePoint:
    """A 3-tuple [x:y:z] representing a point (x/z, y/z) in the projective plane."""
    x: float
    y: float
    z: float
    def __repr__(self):
        return f"({self.x:.3f}: {self.y:.3f}: {self.z:.3f})"
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

### animation aid

"""An isotopy is a function I: (x, y, z, t, dt) -> (x', y', z', t + dt) for all
	-t < dt < 1 - t. It restricts to a homotopy H(x, y, z, t) = I(x, y, z, 0, t)."""
IsotopyFn = Callable[[float, float, float, float, float], tuple[float, float, float]]

### Types which specify wavefronts

"""A function which takes a radius r as input, and returns the minimum and maximum
values of theta on the arc at distance r from the focus and in the same plane region
as the focus """
BoundsFn = Callable[[float], tuple[float, float]]

class Arc:
    """An Arc is specified by a center point, radius, starting angle, and ending angle."""
    center: tuple[float, float]
    radius: float
    angle_bounds: tuple[float, float]

class Segment:
    """A Segment is specified by two endpoints."""
    start: tuple[float, float]
    end: tuple[float, float]

class ArcEnvelope:
    """An ArcEnvelope is a function which takes a radius value as input and produces angle bounds
    as output. It defines a region containing the center point as the union of arcs over
    all positive radii."""
    center: tuple[float, float]
    bounds: BoundsFn

    def interpolate_arcs(self, radius_1: float, radius_2: float) -> Callable[[float, float], float]:
        """Given an initial and final radius, defines a function which isotopes a point
        at angle θ on the first arc forward in time by t.
        """
        min_angle, max_angle = self.bounds(radius_1)
        def f(theta, t):
            alpha = (theta - min_angle) / (max_angle - min_angle)
            new_radius = t * radius_2 + (1 - t) * radius_1
            new_min_angle, new_max_angle = self.bounds(new_radius)
            return alpha * new_max_angle + (1 - alpha) * new_min_angle
        return f
    
    def isotopy(self, radius_1: float, radius_2: float) -> IsotopyFn:
        """Converts the function interpolate_arcs into an isotopy in Cartesian coordinates."""
        def istpy(x: float, y: float, z: float, t: float, dt: float):
            # Convert to polar coordinates (r, θ)
            vec = np.array([x - self.center[0], y - self.center[1]])
            r = np.linalg.norm(vec)

            # Arcsin gives a value in [-π/2, π/2], so we must reflect over the y-axis
            # if the original vector had negative x-coordinate.
            theta = (np.pi - np.arcsin(vec[1] / r)) if vec[0] < 0 else np.arcsin(vec[1] / r)

            # Get current arc and distance along it
            min_angle, max_angle = self.bounds(r)
            alpha = (theta - min_angle) / (max_angle - min_angle)
            
            # Get the corresponding point on the target arc
            target_radius = radius_1 * (1 - (t + dt)) + radius_2 * (t + dt)
            new_min_angle, new_max_angle = self.bounds(target_radius)
            new_theta = alpha * new_max_angle + (1 - alpha) * new_min_angle

            # Convert back to Cartesian coordinates
            return (
                target_radius * np.cos(new_theta) + self.center[0],
                target_radius * np.sin(new_theta) + self.center[1],
                z,
                t + dt
            )
        return istpy
    
class SegmentEnvelope:
    """Same as an ArcEnvelope but defined by two orthogonal lines L1 and L2, where
    the radius is replaced by distance from L1, and angles are replaced by distance from L2."""
    def interpolate_segments(self, radius_1: float, radius_2: float) -> Callable[[float, float], float]:
        pass
    
    def isotopy(self, radius_1: float, radius_2: float) -> IsotopyFn:
        pass

### Types for specifying a conic section

class PolarConicEquation:
    """An equation for r in terms of θ of the form r = C / (1/E + cos(θ - θ_0)).
    Invariant under sending E -> -E, C -> -C, and θ_0 -> θ_0 + π, so we assume
    that E > 0."""
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
        assert e > 0, "Eccentricity must be positive."
        
        self.focus = focus
        self.e = e
        self.c = c
        self.theta_0 = theta_0
        self.other_focus = self._calculate_other_focus()

    def __eq__(self, other: PolarConicEquation):
        case_1 = (self.focus == other.focus) and isclose(self.e, other.e) and isclose(self.c, other.c) and isclose(self.theta_0 - other.theta_0 % (2 * np.pi), 0)
        case_2 = (self.focus == other.focus) and isclose(self.e, -other.e) and isclose(self.c, -other.c) and isclose(self.theta_0 - other.theta_0 % (2 * np.pi), np.pi)
        return case_1 or case_2
    
    def __repr__(self):
        return f"Focus={self.focus}, Eccentricity={self.e}, Scale={self.c}, Angle={self.theta_0}"

    def _calculate_other_focus(self) -> ProjectivePoint:
        """Calculates the position of the other focus from the given data"""
        if self.e == 1:
            return ProjectivePoint(np.cos(self.theta_0), np.sin(self.theta_0), 0)
        else:
            # Signed distance between foci
            dist = self.c * (1/(1 + 1/self.e) - 1/(-1 + 1/self.e))
            return ProjectivePoint(
                x=dist * np.cos(self.theta_0),
                y=dist * np.sin(self.theta_0),
                z=1
            )

    def bounds(self) -> BoundsFn:
        """Returns a function which takes radii r as input, and outputs the minimum and
        maximum values of theta on the arc at distance r from the focus and in the same
        plane region as the focus."""
        # TODO
        if self.c > 0:
            def f(radius: float):
                # Solve the defining equation for theta
                angle = np.arccos(self.c / radius - 1 / self.e)
                # Setting θ = θ_0 yields the smallest possible radius, so the arc must be
                # centered on θ_0 + π if C > 0, and on θ_0 if C < 0.
                return  (self.theta_0 - angle + np.pi, self.theta_0 + angle + np.pi)
        else:
            def f(radius: float):
                # Solve the defining equation for theta
                angle = np.arccos(self.c / radius - 1 / self.e)
                # Setting θ = θ_0 yields the smallest possible radius, so the arc must be
                # centered on θ_0 + π if C > 0, and on θ_0 if C < 0.
                return  (self.theta_0 - angle, self.theta_0 + angle)

        return f

    def to_cartesian(self) -> CartesianConicEquation:
        """Generates the Cartesian form."""
        cart_eq = CartesianConicEquation(
            c_xx = (1 / self.e ** 2) - 1,
            c_yy = (1 / self.e ** 2),
            c_xy = 0.,
            c_x = 2 * self.c,
            c_y = 0,
            c_0 = -self.c ** 2
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
        length = np.sqrt(a**2 - b**2)
        return PolarConicEquation(
            focus=Point2D(x=length, y=0),
            e=length/a,
            c=(b**2)/length,
            theta_0=0
            )
    
    @classmethod
    def std_hyperbola(cls, a: float, b: float) -> PolarConicEquation:
        """Produces the polar form of the hyperbola (x/a)^2 - (y/b)^2 = 1."""
        length = np.sqrt(a**2 + b**2)
        return PolarConicEquation(
            focus=Point2D(x=-length, y=0),
            e=length/a,
            c=(b**2)/length,
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

    def tangent(self, point: Point) -> Hyperplane:
        """Produces the tangent hyperplane at the given point on the curve."""
		# 0 = dP(x, y) = (2Ax + By + D)dx + (Bx + 2Cy + E)dy
        normal = Vector(
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

class Conic:
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
        self.other_focus = polar_eq.other_focus()

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
    def from_cartesian(cls, cart_eq: CartesianConicEquation) -> Conic:
        """Initializes a conic section from a two-variable quadratic equation."""
        Conic(polar_eq=cart_eq.to_polar(), cart_eq=cart_eq)

    @classmethod
    def from_polar(cls, polar_eq: PolarConicEquation) -> Conic:
        """Given one focus, the conic may be defined by a function of polar coordinates
        (r, θ) centered around that focus of the form
        r = C / (1/E + cos(θ - θ_0))"""
        Conic(polar_eq=polar_eq, cart_eq=polar_eq.to_cartesian())

    ### For animating wavefronts

    def make_arc_from_main_focus(self, r: float) -> Arc:
        # TODO Move this to PolarConicEquation?
        """Given a distance r from the main focus, calculates angles (θ1, θ2) such that
        the arc {(r, θ): θ1 < θ < θ2} has endpoints on the conic and lies inside the
        section containing the focus. Then generates the Arc
        
        Used to animate spherical wavefronts centered around the main focus."""
        # TODO
        pass

    def make_arc_from_other_focus(self, r: float) -> Arc | Segment:
        # TODO Move this to PolarConicEquation?
        """A spherical wavefront emanating from the main focus reflects off the
        conic section and turns into a wavefront centered on the other focus.
        Given a total travel distance r from the main focus, returns a specification of
        the corresponding arc centered on the other focus with endpoints on the conic.
        
        - If the conic is an ellipse with distance sum d, this can be described by
          polar coordinates {(d - r, θ): θ1 < θ < θ2} centered at the other focus.
        - If the conic is a hyperbola with distance difference d, this can be described by
          polar coordinates {(d + r, θ): θ1 < θ < θ2} centered at the other focus.
        - If the conic is a parabola, this can be described by a line segment at
          distance r from the directrix and endpoints on the parabola.

        Used to animate spherical wavefronts centered around the other focus."""
        # TODO
        pass

# Sample code to generate a propagating wavefront scene for an ellipse.
if __name__ == "__main__":
    # Parabola with focus at (0, 0) and focal length 1
    polar_eq = PolarConicEquation(Point2D(0, 0), 1, 2, 0)
    cart_eq = polar_eq.to_cartesian()
    assert cart_eq == CartesianConicEquation(0., 0., 1., 4., 0., -4.)
    new_polar_eq = cart_eq.to_polar()
    assert polar_eq == new_polar_eq

    # Standard ellipse
    polar_eq = PolarConicEquation.std_ellipse(5.0, 3.0)
    cart_eq = CartesianConicEquation.std_ellipse(5.0, 3.0)
    assert cart_eq == polar_eq.to_cartesian()

    # Standard hyperbola
    polar_eq = PolarConicEquation.std_hyperbola(5.0, 3.0)
    cart_eq = CartesianConicEquation.std_hyperbola(5.0, 3.0)
    assert cart_eq == polar_eq.to_cartesian()



    # class Isotopy(m.Homotopy):
    #     def __init__(self, isotopy: IsotopyFn, run_time: float = 3, **kwargs):
    #         self.isotopy = isotopy
    #         # Keep a copy of initialization kwargs for re-initialization
    #         self.kwargs = kwargs
    #         def homotopy(x, y, z, t):
    #             x1, y1, z1, t1 = isotopy(x, y, z, 0, t)
    #             return x1, y1, z1
    #         super().__init__(homotopy=homotopy, run_time=run_time, **kwargs)

    # import manim as m
    # from symphony import (Symphony, Sequence, AnimationEvent, Add, Remove)
    # # Ellipse centered at (0, 0) with radii 5 and 3
    # cart_eq = CartesianConicEquation(c_xx=1/25, c_yy=1/16, c_0=1.0)
    # conic = Conic.from_cartesian(cart_eq)
    # main_focus = conic.focus
    # other_focus = conic.other_focus

    # assert main_focus.x == 4.0
    # assert main_focus.y == 0.0
    # assert other_focus.x / other_focus.z == -4.0
    # assert other_focus.y / other_focus.z == 0.0

    # # Polar equations defined around the main focus and other focus.
    # polar_eq_main = conic.polar_eq
    # polar_eq_other = PolarConicEquation(other_focus, polar_eq_main.e, polar_eq_main.c, np.pi + polar_eq_main.theta_0)

    # # Make the envelope corresponding to each focus
    # main_envelope = ArcEnvelope(
    #     center=main_focus,
    #     bounds=polar_eq_main.bounds()
    #     )
    # other_envelope = ArcEnvelope(
    #     center = other_focus.to_cartesian(),
    #     bounds=polar_eq_other.bounds()
    # )

    # # Make isotopies for animation
    # # TODO Make several here.
    # r1 = 0.05
    # start_angle, stop_angle = main_envelope.bounds(r1)
    # arc1 = m.Arc(
    #     arc_center=tuple(*main_envelope.center, 0),
    #     radius=r1,
    #     start_angle=start_angle,
    #     angle=stop_angle - start_angle
    #     )
    # i1 = Isotopy(
    #     isotopy=main_envelope.isotopy(0.05, 8.95),
    #     mobject=arc1,
    #     rate_func=m.linear,
    #     run_time=8.9
    #     )
    
    # r2 = 8.95
    # start_angle, stop_angle = other_envelope.bounds(r2)
    # arc2 = m.Arc(
    #     arc_center=tuple(*other_envelope.center, 0),
    #     radius=r2,
    #     start_angle=start_angle,
    #     angle=stop_angle - start_angle
    #     )
    # i2 = Isotopy(
    #     isotopy=other_envelope.isotopy(8.95, 0.05),
    #     mobject=arc2,
    #     rate_func=m.linear,
    #     run_time=8.9
    #     )
    
    # # Play them simultaneously
    # sequences = []
    # sequences.append([AnimationEvent(
    #     header=[Add(arc1)],
    #     middle=i1,
    #     footer=[Remove(arc1)]
    # )])
    # sequences.append([AnimationEvent(
    #     header=[Add(arc2)],
    #     middle=i2,
    #     footer=[Remove(arc2)]
    # )])

    # symphony = Symphony(sequences).animate()
    
# TODO Test all functions in the polar and Cartesian form.
# def test_1():
#     polar_eq = PolarConicEquation(Point2D(0, 0), 1, 1, 0)
#     cart_eq = polar_eq.to_cartesian()
#     new_polar_eq = cart_eq.to_polar()
#     assert polar_eq == new_polar_eq