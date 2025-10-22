"""
Animations concerning conic sections, of two flavors:
- Animate several ray trajectories emanating from one focus, bouncing off the conic section,
  and then converging on the other focus.
- Animating spherical wavefronts emanating from one focus, bouncing off the conic section,
  and then converging on the other focus."""

from __future__ import annotations
import manim as m
import numpy as np
from lib import (
    Isotopy,
    Symphony, Sequence, AnimationEvent, Add, Remove,
    ConicSection, PolarConicEquation, ArcEnvelope,
    Vector3D, Point3D,
    animate_trajectory
    )

RAY_WIDTH = 1.0
RAY_OPACITY = 1.0
RAY_COLOR = m.WHITE

# We incorporate a small delay between when a wavefront strikes a mirror and when it is
# re-emitted. This ensures that the worldlines for the two waves don't have events 
# changing over at exactly the same time.
REFLECTION_DELAY = 1e-3


class Parabola(m.Scene):
    """Parent class for parabola scenes."""
    # TODO
    pass

class Ellipse(m.Scene):
    """Parent class for Ellipse scenes."""
    def _set_parameters(self):
        """Sets the parameters of the ellipse."""
        self.major_radius = 3.75
        self.minor_radius = 3.0
        self.half_focal_distance = np.sqrt(self.major_radius**2 - self.minor_radius**2)
        self.center = (0, 0)
        self.angle = 0.

    def _make_conic(self):
        """Adds an ellipse and its two foci to the scene, and sets the ConicSection object."""
        self.add(
            m.Ellipse(width=2 * self.major_radius, height=2 * self.minor_radius, color=m.WHITE),
            m.Dot(np.array([self.half_focal_distance, 0.0, 0.0]), radius=0.08),
            m.Dot(np.array([-self.half_focal_distance, 0.0, 0.0]), radius=0.08),
        )
        self.conic = ConicSection.from_polar(PolarConicEquation.std_ellipse(self.major_radius, self.minor_radius))

    def construct(self):
        raise NotImplementedError

class EllipseTrajectory(Ellipse):
    def construct(self):
        self._set_parameters()
        self._make_conic()

        # Set the directions of trajectories
        # TODO Set them to be even
        num_directions = 20
        w = 1 - 1 / num_directions
        directions = [
            Vector3D(np.sin(theta), -np.cos(theta))
            for theta in np.linspace(-w * np.pi, w * np.pi, num_directions)
        ]
        speed = 2.0

        # Make animations
        sequences = []

        cart_eq = self.conic.cart_eq
        main_focus = self.conic.focus
        
        for direction in directions:
            points = cart_eq.make_trajectory(
                start=Point3D(main_focus.x, main_focus.y, 0),
                direction=direction,
                total_dist=2 * self.major_radius
                )
            seq = animate_trajectory(points, speed)
            sequences.append(seq)

        symphony = Symphony(sequences)
        symphony.animate(self)

    

class EllipseWavefront(Ellipse):
    def construct(self):
        self._set_parameters()
        self._make_conic()
        self._make_envelopes()

        # Make animations
        sequences = []
        # TODO Make this part tweakable
        for delay in np.linspace(0.1, 1.9, 10):
            sequences.extend(self._make_wave(delay))

        symphony = Symphony(sequences)
        symphony.animate(self)

    def _make_envelopes(self):
        """Makes the ArcEnvelopes around the two foci."""
        # Polar equations defined around the two foci
        main_focus = self.conic.focus
        other_focus = self.conic.other_focus.to_cartesian()
        polar_eq_main = self.conic.polar_eq
        polar_eq_other = PolarConicEquation(other_focus, polar_eq_main.e, polar_eq_main.c, np.pi + polar_eq_main.theta_0)

        # Envelopes corresponding to each focus
        self.main_envelope = ArcEnvelope(center=main_focus, bounds=polar_eq_main.bounds())
        self.other_envelope = ArcEnvelope(center = other_focus, bounds=polar_eq_other.bounds())

    def _make_wave(self, delay: float, speed = 3.0) -> list[Sequence]:
        """
        Animates a single spherical wave emanating from the main focus and converging on the second.
        The start time is delayed by a chosen amount."""
        a = self.major_radius
        c = self.half_focal_distance
        
        r1 = 0.01
        start_angle, stop_angle = self.main_envelope.bounds(r1)
        arc1 = m.Arc(
            arc_center=(self.main_envelope.center.x, self.main_envelope.center.y, 0),
            radius=r1,
            start_angle=start_angle,
            angle=stop_angle - start_angle,
            stroke_color=m.BLUE,
            stroke_opacity=0.5
            )
        seq1 = [
            AnimationEvent.wait(delay),
            AnimationEvent(
                header=[Add(arc1)],
                middle=Isotopy(
                    isotopy=self.main_envelope.isotopy(r1, a + c - r1),
                    mobject=arc1,
                    rate_func=m.linear,
                    run_time=(a + c - 2 * r1)/speed
                ),
                footer=[Remove(arc1)]
            ),
        ]

        r2 = a + c - 0.01
        start_angle, stop_angle = self.other_envelope.bounds(r2)
        arc2 = m.Arc(
            arc_center=(self.other_envelope.center.x, self.other_envelope.center.y, 0),
            radius=r2,
            start_angle=start_angle,
            angle=stop_angle - start_angle,
            stroke_color=m.RED,
            stroke_opacity=0.5
            )
        seq2 = [
            AnimationEvent.wait(REFLECTION_DELAY + delay + (a - c)/speed),
            AnimationEvent(
                header=[Add(arc2)],
                middle=Isotopy(
                    isotopy=self.other_envelope.isotopy(r2, a + c - r2),
                    mobject=arc2,
                    rate_func=m.linear,
                    run_time=(2 * r2 - (a + c))/speed
                ),
                footer=[Remove(arc2)]
            ),
        ]

        return [seq1, seq2]

class Hyperbola(m.Scene):
    # TODO
    pass
