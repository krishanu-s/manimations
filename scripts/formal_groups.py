import numpy as np
import manim as m
from lib import (
    Isotopy,
    Symphony, Sequence, AnimationEvent, Add, Remove,
    Point2D,
    ConicSection, PolarConicEquation, ArcEnvelope, SegmentEnvelope,
    Vector3D, Point3D,
    animate_trajectory
    )


class FGScene(m.Scene):
    def construct(self):
        a = 1
        b = 0.5

        polar_eq = PolarConicEquation.std_hyperbola(a, b)
        self.conic = ConicSection.from_polar(polar_eq)

        # Define addition of points on this hyperbola
        zero = Point2D(a, 0)
        def add(p1: Point2D, p2: Point2D):
            # Parametrize the line through (a, 0) parallel to p1p2
            p = p1 - p2
            pa = p.x / a
            pb = p.y / b
            
            # Solve the quadratic equation
            t = (-2 * pa) / (pa ** 2 - pb ** 2)
            return zero + Point2D(t * p.x, t * p.y)
        
        # continuous parametrization of the hyperbola
        def exp(t: float):
            return (a * np.cosh(t), b * np.sinh(t), 0)
        
        def log(p: Point2D):
            # Calculate iteratively?
            pass
        
        # Add the hyperbola graph
        self.add(m.ParametricFunction(exp, t_range=[-5, 5]))

        # TODO Add two points and their sum
        identity = m.Dot(exp(0), radius=0.05, color=m.BLUE)
        self.add(identity)
        p1 = m.Dot(exp(0.1), radius=0.05, color=m.BLUE)
        p2 = m.Dot(exp(0.2), radius=0.05, color=m.BLUE)
        p = m.Dot(exp(0.3), radius=0.05, color=m.BLUE).add_updater(
            lambda mobj: mobj.move_to(add(p1.get_center(), p2.get_center())))

        # Add a moving point according to exponentiation
        pt = m.Dot(exp(0), radius=0.05, color=m.BLUE)
        self.add(pt)
        def updater(mobj, dt):
            x, y, z = mobj.get_center()
            mobj.move_to(((a/b) * dt * y + x, (b/a) * dt * x + y, z))
        pt.add_updater(updater)
        

        pts = [exp(t) for t in np.linspace(0, 2, 11)]
        for p in pts:
            self.play(m.FadeIn(m.Dot(p, radius=0.05, color=m.BLUE), run_time=0.2))


        pass