from __future__ import annotations
from typing import List, Optional
import manim as m
import numpy as np
import math
from ray import Point, Vector, Hyperplane, Ray
from polyfunction import Conic
from symphony import play_in_parallel, Symphony, Sequence, AnimationEvent, Add, Remove, Bookend
from trail import RayObject, make_trail

class TestScene1(m.Scene):
	"""Version one of a basic scene with moving points"""
	# WORKING EXAMPLE
	def make_dot(self, point: Point, radius: float = 0.03):
		"""Make a dot at the given point."""
		return m.Dot(point.coords(), radius=radius, color=m.RED
        )
	
	def construct(self):
		points_list = [
			[
				np.array([0., 0., 0.]),
				np.array([1., 0., 0.]),
				np.array([1., 3., 0.]),
			],
			[
				np.array([0., 0., 0.]),
				np.array([0., 2., 0.]),
				np.array([-2., 2., 0.]),
			],
			[
				np.array([0., 0., 0.]),
				np.array([-2., 0., 0.]),
				np.array([-2., -2., 0.]),
			],
		]

		speed = 1.0
		sequences = []
		for points in points_list:

			# Make the starting point
			now_point: Point = Point.from_array(points[0])
			dot = self.make_dot(now_point, radius=0.03)
			sequence: Sequence = []

			# For each segment of the trajectory...
			for i, pt in enumerate(points[1:]):
				next_point: Point = Point.from_array(pt)
				distance = now_point.distance_to(next_point)

				# Define trail behind
				trail = make_trail(dot, starting_point=now_point.coords())

				# Define movement along the line segment
				segment = m.Line(now_point.coords(), next_point.coords(), stroke_width=1, stroke_opacity=0.3)
				mover_animation = m.MoveAlongPath(
					dot,
					segment,
					run_time=distance/speed,
					rate_func=lambda t: t
				)

				# Construct the animation event
				sequence.append(
					AnimationEvent(
						header=[Add(dot), Add(trail)] if i == 0 else [Add(trail)],
						middle=mover_animation,
						# Remove the trail and convert to a line segment
						footer=[
							Remove(trail),
							Add(segment)
						])
				)

				# Update the current point

				now_point = Point.from_array(pt)
			sequences.append(sequence)
			

		symphony = Symphony(sequences)
		symphony.animate(self)


class TestScene2(m.Scene):
	"""Version one of a basic scene with moving points"""
	# def make_dot(self, point: Point, radius: float = 0.03):
	# 	"""Make a dot at the given point."""
	# 	return m.Dot(point.coords(), radius=radius, color=m.RED
    #     )
	
	def construct(self):
		points_list = [
			[
				np.array([0., 0., 0.]),
				np.array([2., 0., 0.]),
				np.array([2., 2., 0.]),
			],
			[
				np.array([0., 0., 0.]),
				np.array([0., 2., 0.]),
				np.array([-2., 2., 0.]),
			],
			[
				np.array([0., 0., 0.]),
				np.array([-2., 0., 0.]),
				np.array([-2., -2., 0.]),
			],
		]

		def foo(dot: m.Dot, point):
			trail = m.VMobject()
			trail.add_updater(lambda x: x.become(
				m.Line(point, dot, stroke_width=2, stroke_opacity=1.0).set_color(m.ORANGE)
			))
			return trail
		
		speed = 1.0

		# Add points as mobjects
		dots = [m.Dot(points[0], radius=0.05, color=m.RED) for points in points_list]
		self.add(*dots)

		for j in range(2):
			# Add trails behind points
			trails = []
			for i in range(3):
				# points = points_list[i]
				# dot = dots[i]
				trails.append(make_trail(dots[i], points_list[i][j]))
			self.add(*trails)
			print(trails)

			# Add movements of points along the lines
			movers = []
			segments = []
			for i in range(3):
				points = points_list[i]
				dot = dots[i]
				distance = np.linalg.norm(points[j+1] - points[j])
				segment = m.Line(points[j], points[j+1], stroke_width=2.0, stroke_opacity=1.0).set_color(m.ORANGE)
				segments.append(segment)
				movers.append(m.MoveAlongPath(
					dot,
					segment,
					run_time=distance/speed,
					rate_func=lambda t: t
				))
			self.play(*movers)

			# Remove the trails and add segments
			self.remove(*trails)
			self.add(*segments)
		self.wait(1.0)
