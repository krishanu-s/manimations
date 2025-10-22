from __future__ import annotations
import manim as m
import numpy as np
from lib.isotopy import Isotopy
from lib.symphony import (Symphony, AnimationEvent, Add, Remove)
from lib.conic import ConicSection, PolarConicEquation, ArcEnvelope

RAY_WIDTH = 1.0
RAY_OPACITY = 1.0
RAY_COLOR = m.WHITE

class WavefrontScene(m.Scene):
    def make_anim_main(self, delay):
        r1 = 0.01
        start_angle, stop_angle = self.main_envelope.bounds(r1)
        arc1 = m.Arc(
            arc_center=(self.main_envelope.center.x, self.main_envelope.center.y, 0),
            radius=r1,
            start_angle=start_angle,
            angle=stop_angle - start_angle
            )
        i1 = Isotopy(
            isotopy=self.main_envelope.isotopy(r1, self.a + self.c - r1),
            mobject=arc1,
            rate_func=m.linear,
            run_time=(self.a + self.c - 2 * r1)/2
            )
        return [
            AnimationEvent.wait(delay),
            AnimationEvent(
                header=[Add(arc1)],
                middle=i1,
                footer=[Remove(arc1)]
            ),
            AnimationEvent.wait((2 - delay) + (self.a - self.c)/2),
        ]

    def make_anim_other(self, delay):
        r2 = self.a + self.c - 0.01
        start_angle, stop_angle = self.other_envelope.bounds(r2)
        arc2 = m.Arc(
            arc_center=(self.other_envelope.center.x, self.other_envelope.center.y, 0),
            radius=r2,
            start_angle=start_angle,
            angle=stop_angle - start_angle
            )
        i2 = Isotopy(
            isotopy=self.other_envelope.isotopy(r2, self.a + self.c - r2),
            mobject=arc2,
            rate_func=m.linear,
            run_time=(2 * r2 - (self.a + self.c))/2
            )
        return [
            AnimationEvent.wait(delay + (self.a - self.c)/2),
            AnimationEvent(
                header=[Add(arc2)],
                middle=i2,
                footer=[Remove(arc2)]
            ),
            AnimationEvent.wait(2 - delay),
        ]

    def construct(self):
        """Animation of spherical wavefronts bouncing within a conic section."""
        a = 4.0
        b = 3.0
        c = np.sqrt(a**2 - b**2)
        self.a = a
        self.c = c

        # Construct an ellipse
        conic = ConicSection.from_polar(PolarConicEquation.std_ellipse(a, b))
        self.add(m.Ellipse(width=2 * a, height=2 * b, color=m.BLUE))

        # Polar equations defined around the two foci
        main_focus = conic.focus
        other_focus = conic.other_focus
        polar_eq_main = conic.polar_eq
        polar_eq_other = PolarConicEquation(other_focus, polar_eq_main.e, polar_eq_main.c, np.pi + polar_eq_main.theta_0)

        # Envelopes corresponding to each focus
        self.main_envelope = ArcEnvelope(
            center=main_focus,
            bounds=polar_eq_main.bounds()
            )
        self.other_envelope = ArcEnvelope(
            center = other_focus.to_cartesian(),
            bounds=polar_eq_other.bounds()
        )

        # Make animations
        sequences = []

        # Animate arcs emanating from both foci
        for delay in [0.1, 0.4, 0.7, 1.0]:
            sequences.append(self.make_anim_main(delay))
            sequences.append(self.make_anim_other(delay))

        symphony = Symphony(sequences)
        symphony.animate(self)