"""Depict the following ideas:

Logarithms. Why are they useful?

Geometric series
- (1-r) + r(1-r) + r^2(1-r) + ... = 1
- Deduce that 1 + r + r^2 + ... = 1 / (1-r)
- Partial version is 1 + r + r^2 + ... + r^{n-1} = (1-r^n)/(1-r)

Quadrature of a parabola y = x^2
- Goal is to calculate the area from x=0 to x=1. Call this A.
- Quadrature under a curve scales under horizontal and vertical stretches.
  - See this with a linear function: area of a triangle.
- Therefore, the area from x=0 to x=2 is 8A.
- This area can also be divided up into four parts of areas A, 1, 1, A. Thus,
  8A = 2A + 2, so A = 1/3.

Quadrature of y = x^3
- List the three we have so far on the right.
- Do a similar calculation as for y = x^2.
- Suggest the formula for y = x^n. Use the same argument to get
  A_n + sum_{k=0}^{n}A_k * (nCk) = 2^{n+1}A_n
  Then use induction and the fact that (nCk)/(k+1) = ((n+1)C(k+1))/(n+1)
  to obtain A_n = 1/(n+1).

Quadrature of a hyperbola y = 1/x
- The natural logarithm ln(a) as the area under the curve y = 1/x from x=1 to x=a
- ln(a) also equals the area from x=b to x=ab, for any b.
- This implies ln(a) + ln(b) = ln(ab)
- This also implies ln(a^n) = n*ln(a)

How would you calculate the logarithm?
- ln(1+a) is the area under 1 / (1 + x) from 0 to a.
- 1 / (1+x) = 1 - x + x^2 - x^3 + ... ()
- And we already calculated the area under each of these.
- So a - a^2/2 + a^3/3 - ..., so long as a <= 1."""

import math
import numpy as np
import manim as m

class NaturalLogarithm(m.Scene):
    def set_params(self):
        self.xmin, self.xmax = -1.0, 6.0
        self.ymin, self.ymax = -1.0, 6.0
        self.xlength, self.ylength = 8.0, 8.0

    def make_ax(self):
        # Make the axes
        self.ax = m.Axes(
            x_range=(self.xmin, self.xmax),
            y_range=(self.ymin, self.ymax),
            x_length=self.xlength,
            y_length=self.ylength,
            ).set_z_index(10)

        # Make the curve
        self.curve = m.ParametricFunction(
            lambda t: self.ax.coords_to_point(1/math.sqrt(math.tan(t)), math.sqrt(math.tan(t)), 0),
            t_range=(0.02, np.pi/2 - 0.02),
            stroke_width=2.5
            ).set_z_index(10)
        
        # Define parametrization functions
        self.param_2 = lambda s, t: self.ax.coords_to_point(t, s/t, 0)
        self.param = lambda t: self.param_2(1, t)

    def construct(self):
        self.set_params()
        self.make_ax()

        # Add the axis and curve
        self.add(self.ax, self.curve)

        # Parameters
        a = 2.5

        # Animation value trackers
        left = m.ValueTracker(1) # Left x-value
        ratio = m.ValueTracker(a) # Ratio between right and left x-values
        prod = m.ValueTracker(1) # Product of x and y coordinates on the curve

        # Add the area defining ln(a)

        left_point = m.Dot(self.param(1), radius=0.06).set_z_index(10)
        right_point = m.Dot(self.param(a), radius=0.06).set_z_index(10)
        g0 = self.ax.plot(lambda t: 1/t, x_range=[0.05, 10], 
            stroke_width=1.0, stroke_opacity=0.5)
        area = self.ax.get_area(g0, [1, a], color=m.BLUE, opacity=0.5)
        value = m.DecimalNumber(math.log(a))

        self.add(left_point, right_point, area, value, g0)

        # Add updaters
        left_point.add_updater(lambda z: z.move_to(
            self.param_2(prod.get_value(), left.get_value())
            ))
        right_point.add_updater(lambda z: z.move_to(
            self.param_2(prod.get_value(), left.get_value() * ratio.get_value())
            ))
        g0.add_updater(lambda z: z.become(
            self.ax.plot(lambda t: prod.get_value()/t, x_range=[0.05, 10], stroke_width=1.0, stroke_opacity=0.5)
        ))
        area.add_updater(lambda z: z.become(
            self.ax.get_area(
                g0, 
                [left.get_value(), left.get_value() * ratio.get_value()], color=m.BLUE, opacity=0.5
                )
        ))
        value.add_updater(lambda d: d.set_value(prod.get_value() * math.log(ratio.get_value())))

        # Animate ln(a) becoming ln(2a)
        self.play(ratio.animate.set_value(2*a), run_time=1.5)
        self.play(ratio.animate.set_value(a), run_time=1.5)

        # Animate proof that ln(ab) = ln(a) + ln(b)
        b = 1.6
        self.play(prod.animate.set_value(b), left.animate.set_value(b), run_time=1.5, rate_func=m.linear)
        self.play(prod.animate.set_value(1), run_time=1.5, rate_func=m.linear)



class QuadratureScene(m.Scene):
    """Scene which demonstrates geometrically that ∫_0^1 x^n dx = 1/(n+1)."""
    def make_axes(self):
        """Sets scaling parameters and adds axes to the scene."""
        self.c = 1.0
        self.ax = m.Axes(
            x_range=[-0.5, 2.5], y_range=[0, 5],
            x_length=8, y_length=8,
            tips=False)
        self.add(*self.ax)

    def get_formula(self, n: int = 2):
        """Animates the construction of the formula."""
        # Construct the graph
        graph = self.ax.plot(lambda x: self.c * x**n, x_range=[0, 2]).set_z_index(10)

        # Highlight the area of interest
        a = self.ax.get_area(graph, [0, 1], color=m.YELLOW, opacity=0.5).add(m.MathTex("A_2", font_size=20).move_to(self.ax.coords_to_point(0.8, 0.25)))
        self.add(a)

        # Add all of the relevant areas, with small versions of their graphs on the right

        # graphs = [self.ax.plot(lambda _: self.c, x_range=[1, 2])]
        # areas = [self.ax.get_area(
        #     graphs[0],
        #     [1, 2], color=m.RED, opacity=0.5).add(m.MathTex("1", font_size=20).move_to(self.ax.coords_to_point(1.5, 0.5)))]
        # for i in range(n):
        #     # TODO Make this more general
        #     g = self.ax.plot(lambda x: self.c * (2 * x - 1), x_range=[1, 2])
        #     a = self.ax.get_area(g, bounded_graph=graphs[-1], opacity=0.5)
        #     # TODO Make this part more general too.
        #     a.add(m.MathTex("1", font_size=20).move_to(self.ax.coords_to_point(1.5, 1.5)))
      
        g0 = self.ax.plot(lambda _: self.c, x_range=[1, 2])
        g1 = self.ax.plot(lambda x: self.c * (2 * x - 1), x_range=[1, 2])

        a0 = self.ax.get_area(g0, [1, 2], color=m.RED, opacity=0.5).add(m.MathTex("1", font_size=20).move_to(self.ax.coords_to_point(1.5, 0.5)))
        self.play(m.FadeIn(a0, run_time=1.0))

        a1 = self.ax.get_area(g1, [1, 2], bounded_graph=g0, color=m.BLUE, opacity=0.5).add(m.MathTex("1", font_size=20).move_to(self.ax.coords_to_point(1.5, 1.5)))
        self.play(m.FadeIn(a1, run_time=1.0))

        a2 = self.ax.get_area(graph, [1, 2], bounded_graph=g1, color=m.YELLOW, opacity=0.5).add(m.MathTex("A_2", font_size=20).move_to(self.ax.coords_to_point(1.85, 3.0)))
        self.play(m.Transform(
            self.ax.get_area(graph, [0, 1], color=m.YELLOW, opacity=0.5).add(m.MathTex("A_2", font_size=20).move_to(self.ax.coords_to_point(0.8, 0.25))),
            a2,
            run_time=1.0
            ))
        pass
  
    def construct(self):
        
        # Draw the curve y = x^2
        graph = self.ax.plot(lambda x: self.c * x**2, x_range=[-1, 2]).set_z_index(10)
        self.add(graph)

        # Highlight the area of interest
        a = self.ax.get_area(graph, [0, 1], color=m.YELLOW, opacity=0.5).add(m.MathTex("A_2", font_size=20).move_to(self.ax.coords_to_point(0.8, 0.25)))
        self.add(a)

        # TODO Transform to match the area from 0 to 2, and change label to 8A_2.
        # TODO Transform back to original size and label
        # TODO Shift the whole graph 1 unit to the left.

        # (x+1)^2 = x^2 + 2x + 1
        g0 = self.ax.plot(lambda _: self.c, x_range=[1, 2])
        g1 = self.ax.plot(lambda x: self.c * (2 * x - 1), x_range=[1, 2])

        a0 = self.ax.get_area(g0, [1, 2], color=m.RED, opacity=0.5).add(m.MathTex("1", font_size=20).move_to(self.ax.coords_to_point(1.5, 0.5)))
        self.play(m.FadeIn(a0, run_time=1.0))

        a1 = self.ax.get_area(g1, [1, 2], bounded_graph=g0, color=m.BLUE, opacity=0.5).add(m.MathTex("1", font_size=20).move_to(self.ax.coords_to_point(1.5, 1.5)))
        self.play(m.FadeIn(a1, run_time=1.0))

        a2 = self.ax.get_area(graph, [1, 2], bounded_graph=g1, color=m.YELLOW, opacity=0.5).add(m.MathTex("A_2", font_size=20).move_to(self.ax.coords_to_point(1.85, 3.0)))
        self.play(m.Transform(
            self.ax.get_area(graph, [0, 1], color=m.YELLOW, opacity=0.5).add(m.MathTex("A_2", font_size=20).move_to(self.ax.coords_to_point(0.8, 0.25))),
            a2,
            run_time=1.0
            ))


        # Highlight the area under the curve from 0 to 1 in yellow. This can be
        # done with a Axes object, which in turn creates Polygons
        # Draw thin gridlines inside.
        # Continuously homotope this to be the quadrature from 0 to 2.
        # Break this region down further into 4 parts.

class CubicQuadratureScene(m.Scene):
    def construct(self):
        # Same thing for cubic

        # Highlight the area under the curve from 0 to 1 in yellow. This can be
        # done with a Axes object, which in turn creates Polygons
        # Draw thin gridlines inside.
        # Continuously homotope this to be the quadrature from 0 to 2.
        # Break this region down further into 4 parts.
        pass

class QuarticQuadratureScene(m.Scene):
    def construct(self):
        # Same thing for cubic

        # Highlight the area under the curve from 0 to 1 in yellow. This can be
        # done with a Axes object, which in turn creates Polygons
        # Draw thin gridlines inside.
        # Continuously homotope this to be the quadrature from 0 to 2.
        # Break this region down further into 4 parts.
        pass
