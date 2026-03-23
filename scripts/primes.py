"""Classes and functions pertaining to a video on prime numbers. This can be split up later."""

from manim import *
from typing import Tuple, List
import numpy as np

def get_primes(n: int) -> List[int]:
    """Returns a list of all primes from 2 to n, using the Sieve of Erastosthenes"""
    # Keep track of which numbers are composite
    composite = {i: False for i in range(2, n+1)}
    primes = []
    for i in range(2, n+1):
        # Pass if composite
        if composite[i]:
            pass
        # Otherwise, perform sieve
        else:
            primes.append(i)
            composite.update({i * j: True for j in range(2, 1 + n // i)})

    return primes

class NumberListScene(Scene):
    """Does the following operations:
    (0) Display title: "What are prime numbers?"
    (1) Fade-in a list of the integers from 1 to n on a single line, and then ... after that.
    (2) Simultaneously colors the composite numbers blue, and the prime numbers red.
    (3) Below each composite number, writes its prime factorization.
    (4) Writes below, "Unique Factorization Theorem: Every positive integer can be decomposed as
        a product of \\textcolor{red}{prime} numbers in exactly one way."
    (5) Removes the prime factorizations and UFT.
    (6) Zooms out the number list, and quickly extends to some larger N (distributed over several rows)
        with the composite numbers colored blue, and the prime numbers red.
    """
    # TODO The "construct" function is too big. Break it down into pieces.
    def construct(self):
        # Length of first list.
        n = 13
        values = range(1, n+1)
        primes = set(get_primes(n))
        composites = set(range(2, n+1)) - primes
        factorizations = {
            4: '2 \\times 2',
            6: '2 \\times 3',
            8: '2 \\times 2 \\times 2',
            9: '3 \\times 3',
            10: '2 \\times 5',
            12: '2 \\times 2 \\times 3'
        }

        # Positions of the number i
        def pos(i):
            return np.array((i - (n + 2) / 2, 0, 0))

        ### (1) write the list of numbers
        values_text = []
        for i in values:
            text = Text(f"{i}").move_to(pos(i))
            values_text.append(text)
            self.play(Write(text, run_time=0.12))
        ellipsis = MathTex('\\ldots').move_to(pos(n+1))
        self.play(Write(ellipsis, run_time=0.15))



        ### (2) Simultaneously color the composite numbers blue, and the prime numbers red.
        transforms = []
        for i in values:
            if i != 1:
                text = values_text[i - 1]
                if i in primes:
                    color = RED
                else:
                    color = BLUE
                colored_text = Text(f"{i}", color=color).move_to(pos(i))
                transforms.append(Transform(text, colored_text))
        self.play(*transforms)

        ### (3) Below each composite number, write its prime factorization in smaller font.
        f_list = []
        for i in composites:
            text = values_text[i - 1]
            f_list.append(MathTex(factorizations[i], font_size=24).next_to(text, DOWN))
        for f in f_list:
            self.play(Write(f, run_time=0.3))

        ### (4) Write the Unique Factorization Theorem.
        uft = MathTex(
            "\\textbf{Unique Factorization Theorem:}",
            font_size=36
        ).next_to(values_text[0], np.array((1., -3., 0.)))
        uft_text = MathTex(
            "\\text{Every positive integer can be decomposed \n as a product of prime numbers in exactly one way.}",
            font_size=30
        ).next_to(values_text[0], np.array((1., -5., 0.)))
        self.play(FadeIn(uft))
        self.play(FadeIn(uft_text))


        ### (5) Fade out factorizations and UFT
        self.play(FadeOut(*f_list, uft, uft_text, ellipsis))

        ### (6) Zooms out the number list, and quickly extends to some larger N (distributed over several rows)
        ### with the composite numbers colored blue, and the prime numbers red.

        w = 30  # number of integers of each line
        m = 210  # total number of integers
        values = range(1, m + 1)
        primes = get_primes(m)
        colors = {i: BLUE for i in range(2, m+1)}
        colors.update({p: RED for p in primes})

        # New text generator for number i
        top_left = np.array((-6.2, 3.2, 0))
        def new_pos(i):
            return top_left + np.array((0.42 * ((i - 1) % w), - 0.42 *((i - 1) // w), 0))

        def new_text(i):
            return Text(f"{i}", font_size=12, color=colors.get(i)).move_to(new_pos(i))

        # Transition to the longer list
        self.play(Transform(values_text[i - 1], new_text(i)) for i in range(1, n + 1))
        values_text.extend([new_text(i)for i in range(n + 1, m + 1)])
        self.play(FadeIn(*values_text[n:]))

        ### (7) Highlight some gaps of length 10. Highlight some twin primes.
        # Flash up the question, "How densely are the prime numbers spaced?"

        ### (8) Write the Prime Number Theorem: "The number of primes less than N is approximately
        ### equal to N / log(N)." Then add a little pointer to the log saying "base e"
        pnt = MathTex(
            "\\textbf{Prime Number Theorem:}",
            font_size=36
        ).next_to(values_text[-w], np.array((1., -2., 0.)))
        pnt_text = MathTex(
            '\\text{The number of primes less than }',
            'N',
            '\\text{ is approximately }',
            '\\frac{N}{\\log(N)}.',
            font_size=30
        ).next_to(values_text[-w], np.array((1., -4., 0.)))
        self.play(FadeIn(pnt, pnt_text))
        self.play(FadeOut(pnt, pnt_text))

        ### (9) Make a table with numerical data.
        table = MobjectTable(
            table=[
                ['N', '\\pi(N)', '\\log_{10}(N)'],
                ['10', '4', '1'],
                ['100', '25', '2'],
                ['1,000', '168', '3'],
                ['10,000', '1,229', '4']
            ],
            element_to_mobject=lambda m: MathTex(m, font_size=24),
            v_buff=0.5,
            h_buff=0.5,
        ).next_to(values_text[-w], np.array((1., -1., 0.)))
        self.play(FadeIn(table))

        # Add a column pertaining to the natural log
        new_table = MobjectTable(
            table=[
                ['N', '\\pi(N)', '\\log_{10}(N)', '\\log(N)'],
                ['10', '4', '1', '2.302'],
                ['100', '25', '2', '4.605'],
                ['1,000', '168', '3', '6.908'],
                ['10,000', '1,229', '4', '9.210']
            ],
            element_to_mobject=lambda m: MathTex(m, font_size=24),
            v_buff=0.5,
            h_buff=0.5,
        ).next_to(values_text[-w], np.array((1., -1., 0.)))
        self.play(FadeOut(table), FadeIn(new_table))
        table = new_table

        # Add a column with the PNT estimate
        new_table = MobjectTable(
            table=[
                ['N', '\\pi(N)', '\\log_{10}(N)', '\\log(N)', 'N / \\log(N)'],
                ['10', '4', '1', '2.302', '4.3'],
                ['100', '25', '2', '4.605', '21.7'],
                ['1,000', '168', '3', '6.908', '144.8'],
                ['10,000', '1,229', '4', '9.210', '1085.7']
            ],
            element_to_mobject=lambda m: MathTex(m, font_size=24),
            v_buff=0.5,
            h_buff=0.5,
        ).next_to(values_text[-w], np.array((1., -1., 0.)))
        self.play(FadeOut(table), FadeIn(new_table))
        table = new_table

        ### (10) Fade out the long number list, while shifting the table up. Then extend the table a few more rows.
        # TODO Figure out how to extend the table.
        self.play(FadeOut(*values_text), table.animate.shift(np.array((0., 3.0, 0.))))

        new_table = MobjectTable(
            table=[
                ['N', '\\pi(N)', '\\log_{10}(N)', '\\log(N)', 'N / \\log(N)'],
                ['10', '4', '1', '2.302', '4.3'],
                ['100', '25', '2', '4.605', '21.7'],
                ['1,000', '168', '3', '6.908', '144.8'],
                ['10,000', '1,229', '4', '9.210', '1085.7'],
                ['100,000', 'foo', 'foo', 'foo', 'foo']
            ],
            element_to_mobject=lambda m: MathTex(m, font_size=24),
            v_buff=0.5,
            h_buff=0.5,
        ).next_to(table, np.array((0., 0., 0.)))
        self.play(FadeOut(table), FadeIn(new_table))
        table = new_table


        ### (11) Transform the Prime Number Theorem to its more accurate form: "The density of the
        ### primes in the neighborhood of N is approximately 1 / log(N)."


def product_of_powers(S: set[int], n: int) -> dict[int, int]:
    """Given a finite 'generating' set S of positive integers > 1, and an upper bound n,
    outputs a dictionary D whose keys i are the integers smaller than n which can be expressed
    as a product of powers of distinct elements of S, and where D[i] equals the number of ways
    this can be done."""
    D = {1: 1}
    for p in S:
        last_update = D
        while len(last_update.keys()) > 0:
            # Get the new terms coming from the next up power of p
            next_update = {
                k * p: v
                for k, v in last_update.items()
                if k * p < n
                }
            last_update = next_update

            # Update D
            old_v = D.get(k, 0)
            for k, v in next_update.items():
                D[k] = v + old_v
    return D

class DensityScene(Scene):
    def construct(self):
        # (1) Restate the Fundamental Theorem of Arithmetic. Then rephrase it as "If you take
        # the set of prime numbers and write down all ways of making products of them, you will
        # get a list which contains every number, exactly once."

        # (2) Draw a number line. Circle 2, and highlight all of its powers.

        # (3) Circle 2 and 3, and highlight all products of their powers.

        # (4) Write the relation between the circled numbers and the highlighted numbers.

        # (5) Switch to logscale.

        # (5) Circle 2, 3, 4, and

        # D
        pass

class ConvolutionScene(Scene):
    def construct(self):
        # TODO
        pass


class NumberLine:
    """A number line object. Holds a list of numbers (floats), as well as numerous display options."""
    # TODO Make this into a subclass of Mobject at some point?
    def __init__(self):
        self.scale_type = "linear"
        self.vals = []
        self.max_val = 0
        self.length = 5

    def _coords(self, val: int):
        # TODO Depends on linear vs log scale
        # TODO Add a leftmost offset.
        return [self.length * (val / self.max_val), 0, 0]

    def _make_dots(self) -> List[Tuple[Dot, Tex]]:
        """Draw the dots corresponding to all of the values"""
        points = []
        for val in self.vals:
            point = Dot(
                point=self._coords(val),
                # TODO Make this size custom-calculated
                radius=0.08,
            )
            label = Tex(f"{val}").next_to(point, DOWN)
            points.append((point, label))
        return points

    def make_display(self):
        points = self._make_dots()
        # TODO Draw the line as a ray with ... at the end.
        line = Line(
            points[0][0].get_center(),
            points[-1][0].get_center()
        )
        return points, line


    def add(self, val: int | None):
        """Adds a number value to the line."""
        self.vals.append(val)
        self.max_val = max(self.max_val, val)

    def to_log_scale(self):
        """Converts the coordinates to log-scale"""
        pass

    def to_linear_scale(self):
        """Converts the coordinates to linear-scale"""
        pass

class NumberLineScene(Scene):
    """A number line which"""
    def construct(self):
        # number of points to be drawn
        n = 10
        values = range(1, n+1)

        # add values to NumberLine
        line = NumberLine()
        for i in values:
            line.add(val=i)

        # Draw them on-screen
        points, line = line.make_display()

        for p, l in points:
            self.add(p, l)
        self.add(line)


        # convert to log-scale

        # circle = Circle()  # create a circle
        # circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        # self.play(Create(circle))  # show the circle on screen
