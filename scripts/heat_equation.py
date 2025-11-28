"""Animation of the heat equation

The Laplacian operator.

Random walks"""

from lib import AutonomousFirstOrderDiffEqSolver, AutonomousSecondOrderDiffEqSolver

class OneDimHeatEquation:
    """Internal value is stored as a 1D array."""
    def diff_eq(self):
        """(d/dt) f(x, t) = -C * (d/dx)^2 f(x, t)"""
        # Calculate (d/dx)^2 at x_n using (f(x_{n-1}) - 2f(x_n) + f_(x_{n+1})) / (Δx)^2

        pass

class OneDimWaveEquation:
    def diff_eq(self):
        """(d/dt)^2 f(x, t) = -C * (d/dx)^2 f(x, t)"""
        pass