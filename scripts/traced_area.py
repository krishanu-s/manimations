"""Stub file on the area traced by a fixed point on a fixed segment pushed along a path in the plane."""

import math
import numpy as np
import manim as m
from lib import interpolate_vals, SmoothClosedPathBezierHandleCalculator

class AreaScene(m.Scene):
    def _make_segment(self, a: float):
        seg_pt_0 = m.Dot(np.array([-1., 0., 0.]))
        seg_pt_1 = m.Dot(np.array([1., 0., 0.]))
        seg_pt_mid = m.Dot((1-a) * seg_pt_0.get_center() + a * seg_pt_1.get_center(), color=m.RED)
        seg_pt_mid.add_updater(lambda mobj: mobj.move_to(
            (1-a) * seg_pt_0.get_center() + a * seg_pt_1.get_center()
        ))
        seg_line = m.Line(seg_pt_0, seg_pt_1, stroke_width=2.0)
        seg_line.add_updater(lambda mobj: mobj.put_start_and_end_on(
            seg_pt_0.get_center(), seg_pt_1.get_center()
        ))
        self.play(m.Create(seg_pt_0), m.Create(seg_pt_1))
        self.play(m.Create(seg_line))
        self.play(m.Wait(1.0))
        self.play(m.Create(seg_pt_mid))
        self.play(m.Wait(1.0))
        
        seg_pt_0.generate_target()
        seg_pt_0.target.move_to(np.array([-3, 3, 0]))
        seg_pt_1.generate_target()
        seg_pt_1.target.move_to(np.array([-2, 3, 0]))
        # TODO Change radius
        self.play(m.MoveToTarget(seg_pt_0), m.MoveToTarget(seg_pt_1))

    def traced_area_problem_definition(self):
        """Defines the traced area problem"""
        self.clear()
        a = 1/3
        self._make_segment(a)

        # Define parametrization f of the curve
        scale = 2.0
        def param(t: float | np.ndarray):
            if isinstance(t, float):
                return -scale * np.stack([np.cos(m.TAU * t), np.sin(m.TAU * t), 0])
            else:
                return -scale * np.stack([np.cos(m.TAU * t), np.sin(m.TAU * t), np.zeros_like(t)], axis=-1)
        def dist(t0: float, t1: float):
            # return np.linalg.norm(param(t0) - param(t1))
            return scale * np.sqrt(
                (np.cos(m.TAU * t1) - np.cos(m.TAU * t0)) ** 2 + (np.sin(m.TAU * t1) - np.sin(m.TAU * t0)) ** 2
                )
        def dist_1(t0: float, t1: float):
            # Derivative of distance with respect to t1
            d = dist(t0, t1) / scale
            return scale * m.TAU * (-np.sin(m.TAU * t1) * (np.cos(m.TAU * t1) - np.cos(m.TAU * t0)) + np.cos(m.TAU * t1) * (np.sin(m.TAU * t1) - np.sin(m.TAU * t0))) / d

        
        # Given t0, finds t1 such that d(f(t0), f(t1)) = 1
        def find_point_at_distance(t0: float, t: float | None = None, d: float = 1.0):
            # Initial guess
            if t is None:
                t = t0 + 0.2
            # Newton's method
            for _ in range(5):
                t -= (dist(t0, t) - d) / dist_1(t0, t)
            return t
        
        num_steps = 20
        p_in_values = np.linspace(0, 1, num_steps + 1)
        q_in_values = np.array([find_point_at_distance(p, t=None, d=2) for p in p_in_values])
        for p, q in zip(p_in_values, q_in_values):
            print(p, q)

        p_out_values = param(p_in_values)
        q_out_values = param(q_in_values)
        r_out_values = (1-a) * p_out_values + a * q_out_values

        curve = m.ParametricFunction(param, t_range=(0, 1))
        self.play(m.Create(curve))

        time = m.ValueTracker(0)

        # Define independent end of segment
        p = m.Dot(param(0))
        # TODO Do interpolation to p_values
        p.add_updater(lambda mobj: mobj.move_to(param(time.get_value())))

        # Define dependent end of segment
        # TODO Add a more general updater to q
        q = m.Dot(param(2 * np.arcsin(1/4) / m.TAU))

        # TODO Interpolate to q_values
        q.add_updater(lambda mobj: mobj.move_to(
            param(interpolate_vals(q_in_values, 0, 1 / num_steps, time.get_value()))
            ))

        # Define segment connecting them
        seg = m.Line(p, q, stroke_width=2.0)
        seg.add_updater(lambda mobj: mobj.put_start_and_end_on(
            p.get_center(), q.get_center()
        ))

        # Define point on segment
        r = m.Dot((1-a) * p.get_center() + a * q.get_center(), color=m.RED)
        r.add_updater(lambda mobj: mobj.move_to(
            (1-a) * p.get_center() + a * q.get_center()
        ))

        # Have it trace out a curve:
        # - Get the value of the time parameter. Accordingly decide how many points to include.
        # - Include the first N r-points, plus one more interpolated point for the remaining fractional part.
        def updater(mobj):
            t = time.get_value()
            n = int(np.floor(t * num_steps))
            points = list(r_out_values[:n + 1])
            if t * num_steps - n > 0:
                r_last = (1 - a) * param(interpolate_vals(p_in_values, 0, 1 / num_steps, t)) + a * param(interpolate_vals(q_in_values, 0, 1 / num_steps, t))
                points += [r_last]
            
            # TODO Turn this into a smooth Bezier curve somehow
            # mobj.become(m.Polygram(*points, color=m.RED).make_smooth())
            mobj.become(m.ArcPolygon(*points, color=m.RED).make_smooth())

        curve = m.VMobject().add_updater(updater)
        self.add(curve)

        self.play(m.Create(p), m.Create(q))
        self.play(m.Create(seg))
        self.play(m.Create(r))
        self.play(time.animate.set_value(1.0), rate_func=m.linear, run_time=3.0)

        # self.play(m.FadeIn())

    def definition_of_area(self):
        """Algebraic definition of the area of a curve f: [0, 1] -> R^2 as the inner product <f, \dot{f}> where
        
        <f, g> := 1/2 * \int_{t=0}^{1} f(t) \times g(t) dt
        
        Animate this as follows:
        - Draw a closed curve.
        - Show that it is an embodiment of a function f by animating a point (labeled by t) moving around on an interval of length 1, and meanwhile the output point f(t) moving around on the curve.
        - Fade in the tangent vector f'(t) as well, attached to this point.
        - Fade out the moving point and draw the points + tangent vectors at a number of points on the curve.
        - Fade in the spanned triangles (connecting each point to the next) to depict the total area.
        - Fade in the displacement vectors to the points.
        - Move all of the tangent vectors to one horizontal line, representing a vector field
        - Move all of the displacement vectors to another horizontal line, representing another vector field
        - On a third line, graph a scalar function whose value represents half the (length of the) cross product between the first two vectors, or equivalently equals the (signed) area of the formed triangle.
        - Shade in the area under this curve, and write "=" and then the numerical value in decimal form.

        Finally, animate the changing of the original curve, and see how ALL of the drawn objects vary, i.e.
        - The points on the curve
        - The tangent vectors
        - The displacement vectors
        - The shaded area
        - The two corresponding vector fields
        - The cross-product scalar field
        - The decimal output
        """
        self.clear()

        # First define an example function and its derivative
        # TODO Choose a different function
        n = 10
        origin = np.array([0., 2., 0.])
        anchors = np.stack([origin + np.array([np.cos(m.TAU * k / n), np.sin(m.TAU * k / n), 0]) for k in range(10)], axis=0)
        # anchors = np.stack([
        #     np.array([ 1., 0., 0.]),
        #     np.array([ 1., 1., 0.]),
        #     np.array([ 0., 2., 0.]),
        #     np.array([ 0., 3., 0.]),
        #     np.array([-2., 2., 0.]),
        #     np.array([-3., 2., 0.]),
        #     np.array([-3., 1., 0.]),
        #     np.array([-2.,-1., 0.]),
        #     np.array([ 1.,-2., 0.]),
        #     np.array([ 2.,-1., 0.]),
        # ], axis=0)
        bezier_calculator = SmoothClosedPathBezierHandleCalculator(n)
        bezier_fn, bezier_fn_derivative = bezier_calculator.get_smooth_global_bezier_function(anchors)

        # Animate the creation of the curve
        curve = m.ParametricFunction(bezier_fn, t_range = [0, 1.0001], stroke_width = 2.0)
        self.play(m.Create(curve))

        # Trace a point and its tangent vector traveling around the curve.
        time = m.ValueTracker(0)
        pt = m.Dot(bezier_fn(0), radius=0.08, color=m.RED)
        pt.add_updater(lambda mobj: mobj.move_to(bezier_fn(time.get_value())))

        arrow_scale = 0.05
        vec = m.Arrow(
            bezier_fn(0), bezier_fn(0) + arrow_scale * bezier_fn_derivative(0), color=m.RED,
            max_tip_length_to_length_ratio=0.5, buff=0)
        vec.add_updater(lambda mobj: mobj.put_start_and_end_on(
            bezier_fn(time.get_value()), bezier_fn(time.get_value()) + arrow_scale * bezier_fn_derivative(time.get_value())
        ))
        self.play(m.Create(pt), m.Create(vec))
        self.play(time.animate.set_value(1.0), rate_func=m.linear, run_time=3.0)

        # Leave behind many snapshots
        def make_pt_and_vec(t: float, color):
            p = m.Dot(bezier_fn(t), radius=0.05, color=color)
            v = m.Arrow(
                bezier_fn(t), bezier_fn(t) + arrow_scale * bezier_fn_derivative(t), color=color,
                max_tip_length_to_length_ratio=0.25, buff=0
                )
            return p, v
        
        num_pts = 15
        colors = m.color_gradient(reference_colors=[m.ORANGE, m.YELLOW, m.GREEN], length_of_output=num_pts)
        points_and_vecs = [make_pt_and_vec(i / num_pts, colors[i]) for i in range(num_pts)]
        points: tuple[m.Dot]
        vecs: tuple[m.Arrow]
        points, vecs = tuple(zip(*points_and_vecs))
        disp_vecs = tuple(m.Arrow(origin, p, buff=0, color=colors[i]) for i, p in enumerate(points))
        self.play(m.FadeIn(*points, *vecs, *disp_vecs))


        # TODO For each snapshot, draw a triangle
        triangles = []
        for i in range(num_pts):
            triangles.append(m.Polygon(
                origin,
                points[i].get_center(),
                points[i].get_center() + 1.5 * vecs[i].get_vector() / (num_pts * arrow_scale)
                # points[i+1].get_center()
            ).set_opacity(0.3).set_stroke(width=1.0))
        # triangles.append(m.Polygon(
        #     m.ORIGIN,
        #     points[num_pts - 1].get_center(),
        #     points[0].get_center()
        # ).set_opacity(0.3).set_stroke(width=1.0))
        self.play(m.FadeIn(m.Dot(origin)))
        for t in triangles:
            self.play(m.FadeIn(t), run_time=5.0 / num_pts)
        
        self.play(m.FadeOut(*triangles))

        # TODO Move these snapshots each onto their own number line, one above the other, representing the two vector fields
        vector_field_1 = m.Line(
            np.array([-4, -1, -10.]),
            np.array([-4, -1, -10.]) + 8.0 * np.array([1., 0., 0.])
            )
        for i, v in enumerate(vecs):
            v.generate_target()
            v.target.move_to(np.array([-4, -1, 0.]) + 8.0 * np.array([i / num_pts, 0., 0.]) + v.get_vector() / 2)
        
        vector_field_2 = m.Line(
            np.array([-4, -2, -10.]),
            np.array([-4, -2, -10.]) + 8.0 * np.array([1., 0., 0.])
            )
        for i, dv in enumerate(disp_vecs):
            dv.generate_target()
            dv.target.move_to(np.array([-4, -2, 0.]) + 8.0 * np.array([i / num_pts, 0., 0.]) + (points[i].get_center() - origin) / 2)

        # TODO Animate the creation of a scalar function whose value at t is equal to the numerical cross product of the derivative vector with the displacement vector. Do this by computing said values pointwise, plotting these, and interpolating a new open Bezier curve.
        cross_product_field = m.Line(
            np.array([-4, -4, -10.]),
            np.array([-4, -4, -10.]) + 8.0 * np.array([1., 0., 0.])
            )
        
        
        self.play(*[m.MoveToTarget(v) for v in vecs])
        self.play(m.FadeIn(vector_field_1, vector_field_2))
        self.play(*[m.Transform(dv, dv.target) for dv in disp_vecs])


    def construct(self):
        # self.traced_area_problem_definition()
        self.definition_of_area()
