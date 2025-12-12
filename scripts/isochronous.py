"""
New version of pendulum.py. Contains the scripting for a video about oscillators and celestial navigation, centering around a discussion of the solution to the isochronous pendulum problem (the tautochrone) by Huygens.
"""

import math
from typing import Callable
import numpy as np
from scipy.special import ellipj
import manim as m
from lib import ParametrizedHomotopy, AutonomousSecondOrderDiffEqSolver, interpolate_vals

def cis(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta), 0])

# class Globe(m.VGroup, metaclass=m.ConvertToOpenGL):
#     def __init__(self, camera: m.Camera):
#         # Retain a pointer to the camera for rendering purposes
#         self.camera = camera
#         pass
#     def add_longitude_line(self, phi: float):
#         # Add a line which is part-normal, part-dashed, and auto-updates based on the camera angle
#         pass
#     def add_latitude_line(self, theta: float):
#         # Add a line which is part-normal, part-dashed, and auto-updates based on the camera angle
#         pass

class IsochronousScene(m.ThreeDScene):
    def pendulum_swing(self):
        self.clear()
        pass

    def longitude_problem(self):
        """A graphical explanation of the longitude problem.
        The main points are that:
        - Location on earth is specified by latitude and longitude.
        - Celestial sphere remains mostly constant, providing an absolute frame against which to place position.
        - If two points are at the same latitude and their longitude differs by 1 degree (i.e. 110 km at the equator), then their respective views of space at times separated by 4 minutes will be the same. 4 minutes is not long enough for the corresponding shift of celestial bodies to make a difference.
        - Therefore, any method of calculating longitude based on astronomical methods will require an accurate measure of time. Every 4 minutes of inaccuracy corresponds to 1 degree of inaccuracy in the longitude.

        (Next step: useful in both mapmaking and sea navigation. Crucial for a nation dependent on long-distance trade, and essentially enabled colonization of the Global South by Western European maritime empires. This is why Britain and Netherlands led the way (then Spain, etc.).
        
        A keen interest at the national and elite level in developing reliable navigation and mapmaking techniques (specifically the longitude problem) drove interest in accurate timekeeping devices, and that is where this problem came from.)
        """
        self.clear()
        camera_phi = 60
        camera_theta = -10
        self.set_camera_orientation(phi=camera_phi * m.DEGREES, theta=camera_theta * m.DEGREES, zoom=2.0)

        # TODO Draw a sphere, and then use it + the camera orientation to determine
        # which parts of the latitude/longitude lines are normal/dashed
        
        # TODO Make this part-normal, part-dashed and linked to the sphere above as a method add
        equator = m.ParametricFunction(
            lambda t: np.array([np.cos(t), np.sin(t), 0]),
            t_range=(0, m.TAU, m.TAU/100),
            stroke_opacity=0.5,
            )
        num_longitude_lines = 12

        # TODO Make this part-normal, part-dashed and linked to the sphere above as a method add
        longitude_lines = []
        for phi in m.TAU * np.linspace(0, 1 - 1/num_longitude_lines, num_longitude_lines):
            longitude_lines.append(m.ParametricFunction(
                lambda t: np.array([np.cos(t) * np.cos(phi), np.cos(t) * np.sin(phi), np.sin(t)]),
                    t_range=(0, m.TAU, m.TAU/100),
                    stroke_opacity=0.5,
                    stroke_width=2.0
            ))
        self.add(equator)
        self.add(*longitude_lines)

        earth = m.Sphere(center=np.array([0., 0., 0.]), radius=1.0, shade_in_3d=False)
        self.add(earth)

        # Rotation
        self.begin_ambient_camera_rotation(rate=0.1)
        self.play(m.Wait(2.0))
        self.stop_ambient_camera_rotation()

    def isochronous_oscillators(self):
        """An oscillating system is `isochronous' if its oscillation period is the same at any amplitude. Show several oscillating systems classified as isochronous or not, as well as their oscillation time as a function of the amplitude.
        Isochronous oscillators are also called `harmonic' oscillators."""
        self.clear()
        # Orbiting plant: not isochronous. Period is distance^(3/2).
        star_position = np.array([0, 0, 0])
        star = m.Dot(star_position, radius=0.14)
        self.add(star)

        g = 4.0
        run_time = 5.0
        Nt = 500
        dt = run_time / Nt
        time = m.ValueTracker(0)

        def make_planet(dist: float):
            # TODO Solve for the trajectory explicitly and plot it
            x0 = np.array([dist, 0, 0])
            v0 = np.array([0, math.sqrt(g / dist), 0])
            # Energy is PE + KE = -mMg/x0 + 1/2 * mv0^2
            solver = AutonomousSecondOrderDiffEqSolver(
                t0=0,
                x0=x0,
                v0=v0,
                f=lambda val: -g * val[0] / (np.linalg.norm(val[0]) ** 3)
                )
            
            results = [x0.copy()]
            for _ in range(Nt):
                solver.step(dt)
                # TODO If necessary, add a conservation of energy step which scales v
                results.append(solver.val[0].copy())
            results = np.stack(results, axis=0)
            planet = m.Dot(np.array([0, 0, 0]) + x0, radius=0.04)
            planet.add_updater(lambda mobj: mobj.move_to(
                star_position + interpolate_vals(results, 0, dt, time.get_value())
            ))
            return planet
        
        planets = [make_planet(dist) for dist in (1.0, 2.0, 3.0)]
        self.add(*planets)
        self.play(time.animate.set_value(run_time), rate_func=m.linear, run_time=run_time)

        # Pendulum: *almost* isochronous. Period is computed via Jacobi elliptic functions
        # Foucault pendulum: ?
        # Vibrating string: isochronous
        # LC circuit: isochronous
        pass

    def tautochrone_solution(self):
        """Present the solution to the tautochrone problem. The transition in:
        - It had already been standard to use series and integrals to do computations in astronomy, but this problem was one of the precursors to calculus of variations"""
        pass

    def test(self):
        # NOTES ON DEPTH-TESTING:
        # - For *still* images, the mobjects are *sorted* according to their depth from the camera angle. This is done in the pathway
        #   ThreeDCamera.capture_mobjects -> ThreeDCamera.get_mobjects_to_display
        # - When animating *movement*, only the moving mobjects are re-rendered. Hence, they are consistently rendered *over* the still mobjects. No depth-testing is done. See:
        #   Scene.play_internal, line 1297
        # TODO Open the above as a bug.
        # - What about when two *different* mobjects are moving? In this case, depth testing is indeed done properly.

        # What's the takeaway?
        # - Break down 3D mobjects into sub-objects, since only whole pieces are rendered.
        # - To ensure no issues with depth, always animate all objects which might overlap in the camera view.
        self.clear()
        camera_phi = 90
        camera_theta = 0
        self.set_camera_orientation(phi=camera_phi * m.DEGREES, theta=camera_theta * m.DEGREES, zoom=1.0)
        s1 = m.Sphere(center=np.array([0., 0., 0.]), radius=1.0, resolution=20)
        s2 = m.Sphere(center=np.array([0., 0., -1.5]), radius=2.0, resolution=20)
        s2.generate_target()
        s2.target.shift(np.array([0., 0., 2.]))
        s1.generate_target()
        s1.target.shift(np.array([0., 0., -2.]))
        self.add(s1, s2)
        self.play(m.MoveToTarget(s2), m.MoveToTarget(s1))


    def construct(self):
        # self.longitude_problem()
        # self.test()
        self.isochronous_oscillators()