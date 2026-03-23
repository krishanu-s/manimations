"""Testing ground"""

import math
import numpy as np
from manim import *
from lib.conic import *
from functools import partial


# Scene describing how to parametrize Pythagorean triples, and integer solutions to any other homogeneous
# quadratic equation, by parametrizing rational points on a conic section.
class PythagoreanTriples(Scene):
    def scene_0(self):
        self.clear()
        pass

    def scene_1(self):
        """Draw a right triangle with sidelengths a, b, and c. Then write in the Pythagorean theorem"""
        self.clear()

        # Draw a Pythagorean triangle
        a = ValueTracker(1)
        b = ValueTracker(1)

        def set_label(mobj: DecimalNumber, val: ValueTracker):
            """Set the label of the decimal number to the value tracker's value"""
            mobj.set_value(val.get_value())
        
        def update_triangle(mobj: Polygon):
            mobj.set_points([
                np.array([0, 0, 0]),
                np.array([0, 1, 0]),
                np.array([1, 0, 0]),
                ])
        
        triangle = Polygon(np.array([0, 0, 0]), np.array([0, a.get_value(), 0]), np.array([b.get_value(), 0, 0]),).set_color(WHITE)
        # triangle.add_updater(update_triangle)
        
        right_angle = Elbow(0.1)

        label_a = DecimalNumber(1, font_size=24).move_to([-0.3, 0.5, 0])
        label_b = DecimalNumber(1, font_size=24).move_to([0.5, -0.3, 0])
        label_c = DecimalNumber(math.sqrt(2), font_size=24).move_to([0.7, 0.7, 0])

        
        # triangle_group = VMobject().add(
        #     triangle,
        #     right_angle,
        #     label_a,
        #     label_b, 
        #     label_c,
        #     )
        # self.add(triangle)
        self.play(Create(triangle, rate_func=linear), Create(right_angle, rate_func=linear))
        self.play(FadeIn(label_a, label_b, label_c))
        self.play(Write(MathTex('a^2 + b^2 = c^2').move_to([0, -1, 0])))

        
        # Change values a, b, and c
        triangle.add_updater(update_triangle)
        label_a.add_updater(partial(set_label, a))
        label_a.add_updater(lambda mobj: mobj.move_to([-0.3, a.get_value()/2, 0]))
        label_b.add_updater(partial(set_label, b))
        label_b.add_updater(lambda mobj: mobj.move_to([b.get_value()/2, -0.3, 0]))
        label_c.add_updater(lambda mobj: mobj.set_value(math.sqrt(a.get_value() ** 2 + b.get_value() ** 2)))
        label_c.add_updater(lambda mobj: mobj.move_to([b.get_value()/2 + 0.2, a.get_value()/2 + 0.2, 0]))

        self.play(a.animate.set_value(2.0))

    def scene_2(self):
        self.clear()
        pass

    def scene_3(self):
        self.clear()
        pass

    def scene_4(self):
        self.clear()
        pass

    def scene_5(self):
        self.clear()

        # # Parameter for the slope of the stereographic projection line
        # m = manim.ValueTracker()
        # m.set_value(-2.0)

        # # Cartesian form of the conic section
        # cart_eq = CartesianConicEquation(1, 0, 1, 0, 0, -1)
        # polar_eq = cart_eq.to_polar()

        # curve = manim.ParametricFunction(function=lambda t: polar_eq.param(t), t_range=(0, manim.TAU,))

        # axes = manim.Axes((-3, 3), (-3, 3))
        # self.play(manim.FadeIn(axes, curve))

        # # Add a temporary value tracker to change the equation
        # c_xy = manim.ValueTracker()
        # def update_curve(c: manim.ParametricFunction) -> manim.ParametricFunction:
        #     cart_eq.c_xy = c_xy.get_value()
        #     polar_eq = cart_eq.to_polar()
        #     return manim.ParametricFunction(function=lambda t: polar_eq.param(t), t_range=(0, manim.TAU,)) 
        # curve.add_updater(update_curve)
        # self.play(c_xy.animate.set_value(1.0))
        # curve.remove_updater(update_curve)
        

    def scene_6(self):
        self.clear()
        pass

    def construct(self):
        self.scene_1()
