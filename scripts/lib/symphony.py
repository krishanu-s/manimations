### Tools to play several sequences of animations in parallel

from __future__ import annotations
from dataclasses import dataclass
import manim as m
from .isotopy import Isotopy

"""The minimum runtime allowed for an AnimationEvent. AnimationEvents with runtime
shorter than this will be deleted."""
EVENT_TIME_TOLERANCE = 1e-6

@dataclass
class Add:
	mobject: m.Mobject
	def __repr__(self):
		return f"Add: {self.mobject}"

@dataclass
class Remove:
	mobject: m.Mobject
	def __repr__(self):
		return f"Remove: {self.mobject}"

"""A Bookend is a list of run-time = 0 events."""
Bookend = list[Add | Remove]

def split(animation: m.Animation, alpha: float) -> tuple[m.Animation, m.Animation]:
	"""Splits an animation into two pieces of lengths alpha * runtime and (1-alpha) * runtime,
	such that their composition yields the original animation."""
	rt = animation.run_time
	if isinstance(animation, m.Wait):
		return m.Wait(alpha * rt), m.Wait((1 - alpha) * rt)
	elif isinstance(animation, m.MoveAlongPath):
		mobj = animation.mobject
		path = animation.path
		assert isinstance(path, m.Line)
		assert animation.rate_func == m.linear
		midpoint = path.point_from_proportion(alpha)
		return (
			m.MoveAlongPath(mobj, m.Line(path.start, midpoint), run_time=alpha * rt, rate_func=m.linear),
			m.MoveAlongPath(mobj, m.Line(midpoint, path.end), run_time=(1 - alpha) * rt, rate_func=m.linear)
		)
	elif isinstance(animation, Isotopy):
		isotopy = animation.isotopy

		# TODO This can be numerically unstable if alpha is very close to 1.
		def first_istpy(x, y, z, t, dt):
			x1, y1, z1, t1 = isotopy(x, y, z, t * alpha, dt * alpha)
			return x1, y1, z1, t1 / alpha
		
		def secnd_istpy(x, y, z, t, dt):
			x1, y1, z1, t1 = isotopy(x, y, z, alpha + t * (1 - alpha), dt * (1 - alpha))
			return x1, y1, z1, (t1 - alpha) / (1 - alpha)
		
		return (
			Isotopy(isotopy=first_istpy, run_time = alpha * rt, **animation.kwargs),
			Isotopy(isotopy=secnd_istpy, run_time = (1 - alpha) * rt, **animation.kwargs)
		)
	pass

class AnimationEvent:
	"""An AnimationEvent is a m.Animation to go into scene.play(), or a m.MObject to go into scene.add() or scene.remove(),
	or a sequence of the above with at most one having runtime > 0.
	We assume that every AnimationEvent has positive total run-time. This is the `middle' section, while the header
	and footer are runtime = 0."""

	def __init__(self,
		header: Bookend,
		middle: m.Animation,
		footer: Bookend
	):
		self.header: Bookend = header
		self.middle: m.Animation = middle
		self.footer: Bookend = footer
		self.run_time: float = self.middle.run_time

	def __repr__(self):
		return f"Header: {self.header}, Middle: {self.middle}, Footer: {self.footer}"

	def play_header(self, scene: m.Scene):
		self.play_bookend(scene, self.header)

	def play_footer(self, scene: m.Scene):
		self.play_bookend(scene, self.footer)

	def play_bookend(self, scene: m.Scene, bookend: Bookend):
		for a in bookend:
			if isinstance(a, Add):
				scene.add(a.mobject)
			elif isinstance(a, Remove):
				scene.remove(a.mobject)

	def play(self, scene: m.Scene):
		"""Play the AnimationEvent in the given scene"""
		self.play_header(scene)
		scene.play(self.middle)
		self.play_footer(scene)

	def split(self, t: float) -> tuple[AnimationEvent, AnimationEvent | None]:
		"""Fragments the AnimationEvent into an initial portion of run-time t
		and a final portion with the remainder of the run-time."""
		if t >= self.middle.run_time - EVENT_TIME_TOLERANCE:
			return self, None
		alpha = t / self.middle.run_time

		# TODO Implement this specifically for each Animation type for which it'd be allowed.
		if isinstance(self.middle, m.Wait):
			first, second = split(self.middle, alpha)
		elif isinstance(self.middle, m.MoveAlongPath):
			first, second = split(self.middle, alpha)
		elif isinstance(self.middle, Isotopy):
			first, second = split(self.middle, alpha)
		else:
			raise InterruptedError

		return (AnimationEvent(self.header, first, []), AnimationEvent([], second, self.footer))

	@classmethod
	def wait(cls, t: float) -> AnimationEvent:
		"""Wraps a `wait' animation."""
		return AnimationEvent([], m.Wait(t), [])


"""A Sequence is a list of AnimationEvent's to be played sequentially."""
Sequence = list[AnimationEvent]
"""A Parallel is a list of AnimationEvent's of the same length to be played in parallel"""
Parallel = list[AnimationEvent]

def play_in_parallel(events: Parallel, scene: m.Scene):
	"""Plays the animation events in parallel"""
	# First play any add/remove steps at the beginning
	for e in events:
		e.play_header(scene)

	# Then play the parts with positive run-time
	scene.play(*[e.middle for e in events])

	# Then play the add/remove steps at the end
	for e in events:
		e.play_footer(scene)

class Symphony:
	"""A Symphony is a list of pairs (Sequence, start_time >= 0), to be played in parallel."""
	sequences: list[Sequence]

	def __init__(self, sequences: list[Sequence]):
		self.sequences = sequences
		self.end_time: float = sum(anim.run_time for anim in sequences[0])

	def animate(self, scene: m.Scene):
		"""
		We animate a Symphony as follows.
		1. Number the Sequences as S_1, S_2, S_3, ..., S_n
		2. For each Sequence, calculate all start times of its SceneEvents.
		   Make a large list SceneEvents of triples (time, Sequence, SceneEvent (referred to by index)), ordered by time.
		3. Set executing = {1: -1, 2: -1, ..., n: -1} and current_time = 0.
		4. For each (t, S_k, j) in SceneEvents:
			a. For i=1, ..., n, we advance the currently-executing SceneEvent in S_i, by (t - current_time).
			   This requires "fragmenting" the SceneEvent. I would expect a version of this already exists
			   in Manim, but I can't find it. Certainly it is not hard to do for linear motion of objects.
			   (See Animation.interpolate() and interpolate_submobject, which respectively modify a mobject
						part of the way to its end state.

			   This can be skipped if t == current_time, and similarly we immediately execute any time-0 events.
		    b. Set executing[k] += 1 and pop off the front element of S_k.
		5. Continue until current_time is greater than the end time of the last SceneEvent.
		"""

		# Make a large list scene_events of all triples (start_time, SequenceIndex, SceneEventIndex), ordered by time.
		# Eliminate any duplicate times from this list.
		# TODO Deal with the situation where two sequences have event changes
		#      at the same time.
		# TODO Switch over to end times instead of start times to ensure everything is executed.
		# TODO Figure out why some "Remove" footers are never getting executed.
		scene_events = []
		times = {}
		for seq_ind, seq in enumerate(self.sequences):
			start_time = 0
			for ev_ind, ev in enumerate(seq):
				if start_time == 0:
					pass
				elif start_time in times.keys():
					pass
				else:
					scene_events.append((start_time, seq_ind, ev_ind))
					times[start_time] = 0
				start_time += ev.run_time
		scene_events.sort(key=lambda x: x[0])

		# Set initial state
		current_time = 0
		executing = {i: seq.pop(0) for i, seq in enumerate(self.sequences)}

		# Proceed through scene_events from the front.
		# If the next event is Event j in Sequence i, and starts at time t...
		for t, i, j in scene_events:
			# The time to elapse forward
			elapsed_time = t - current_time

			# Split all of the currently executing events
			to_execute: list[AnimationEvent] = []
			for ind, event in executing.items():
				if event is None:
					continue
				
				# TODO In some cases, this may result in a very short remainder,
				# which can create numerical instability with isotopies.
				# We should see about handling this case, e.g. if elapsed_time is extremely
				# close to the length of the event
				event_fragment, remainder = event.split(elapsed_time)
				to_execute.append(event_fragment)
				# If this depleted the executing event in sequences[ind], pop off the next one
				if remainder is None:
					next_event = self.sequences[ind].pop(0) if len(self.sequences[ind]) > 0 else None
					executing[ind] = next_event
				else:
					executing[ind] = remainder

			# Play the animation fragments in parallel
			play_in_parallel(to_execute, scene)

			# Move the current time forward
			current_time = t

		# Play the final portion if there are any left
		to_execute = [e for e in executing.values() if e is not None]
		play_in_parallel(to_execute, scene)

if __name__ == "__main__":
	# TODO Add tests.
	pass