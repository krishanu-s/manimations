"""
This is a file of helper functions and associated visualization scripts.

The goal is to explore addition of points on conic sections, parametrization of
conic sections, and exponentiation/logarithms on those conic sections. From there,
we can move to the same for elliptic curves.

First, define point addition.

For a circle    x^2 + y^2 = 1, exponentiation recovers the map t -> ( cos(t),  sin(t)).
For a parabola  x   - y^2 = 0, exponentiation recovers the map t -> (    t^2,       t).
For a hyperbola x^2 - y^2 = 1, exponentiation recovers the map t -> (cosh(t), sinh(t)).

History. Show visual correspondence mapping a circle to a hyperbola, via sundial.
It relies on the fact that hyperbola can be parametrized by (sec(θ), tan(θ)).

There is a "natural logarithm" function when we work over the real numbers,
making point exponentiation (i.e., g -> g^n = exp(n * ln(g)) for some large n) "fast".

For elliptic curves, this happens when we work over the *complex numbers*.

*** DEFINING OVER OTHER FIELDS ***

Consider over the rational numbers as an example. There is no longer a
ln() nor an exp() function.

To calculate logs, we have to use powers and then consider sizing.

Then consider the same over a finite field. Again, there is definitely no exp() nor ln().
The "hardness" of logarithms in this case underlies the RSA cryptosystem.

"""