import numpy as np
import unittest
from gf_analytic_eq import GFeq

class TestGFAnalyticEq(unittest.TestCase):
    
    def verify_gs_solution(self, eps, kappa, delta, nu, tol=1e-4):
        # Initialize equilibrium model
        gf = GFeq(eps=eps, kappa=kappa, delta=delta, nu=nu)
        
        # Calculate the eigenvalue 'alpha'
        alpha = gf.get_alpha(show_plot=False)
        self.assertIsNotNone(alpha)
        self.assertGreater(alpha, 0.0)
        
        # Create a grid to check the GS equation
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-gf.kappa, gf.kappa, 50)
        X, Y = np.meshgrid(x, y)
        
        # Check if the solution is a valid Grad-Shafranov solution
        err = gf.check_GS(X, Y, alpha)
        max_err = np.amax(err)
        
        # The error should be very small
        self.assertLess(max_err, tol, f"GS error too large: {max_err} >= {tol}")

    def test_circular_plasma(self):
        """Test a standard circular cross-section."""
        self.verify_gs_solution(eps=0.33, kappa=1.0, delta=0.0, nu=1.0)

    def test_d_shaped_plasma(self):
        """Test a typical elongated D-shaped cross-section."""
        self.verify_gs_solution(eps=0.33, kappa=1.6, delta=0.3, nu=1.0)

    def test_high_nu_plasma(self):
        """Test a configuration with higher nu (related to poloidal beta)."""
        self.verify_gs_solution(eps=0.33, kappa=1.5, delta=0.2, nu=2.0)

if __name__ == '__main__':
    unittest.main()
