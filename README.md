# Guazzotto-Freidberg Analytic Tokamak Equilibrium
Code to calculate analytic solutions to the Grad–Shafranov equation using the Guazzotto-Freidberg method [1].

## Description
This program produces analytic solutions to the Grad–Shafranov equation describing tokamak equilibria using a method developed by Guazzotto and Freidberg in [1].

In this program we create up/down symmetric equilibria with model flux surfaces based on the Miller geometry [2]. 
Flux surface shapes are characterized by scalars:
- ``eps`` = a/R0 (inverse aspect ratio)
- ``kappa`` (elongation)
- ``delta`` (triangularity)
- ``nu`` is related to the poloidal beta.

The separation of variables method outlined in [1] restricts the 'free functions' -- plasma pressure, p, and ff' -- to quadratic functions of the poloidal magnetic flux, \psi. 

To compute the lab-frame poloidal magnetic flux, \Psi(R, Z), three additional parameters are required: 
- ``B0`` the vacuum toroidal field  
- ``p0`` the on-axis plasma pressure
- ``R0`` the axial major radius
Values with SI units ([T], [Pa], and [m]) are expected.

## Examples
Several standard cases are provided at the bottom of the main program.
```py
cases = {
  "circle" : dict(eps=0.33,kappa=1,delta=0,nu=1),
  "ellipse" : dict(eps=0.25,kappa=2,delta=0,nu=1),
  "D" : dict(eps=0.33, kappa=1.8, delta=0.4, nu=0.3),
  "negD" : dict(eps=0.33, kappa=1.9, delta=-0.6, nu=0.5),
}
```
What follows is a more detailed example of the "circle" case.
We create an instance of the Guazzotto-Freidberg equlibrium class (``GFeq``) and evaluate the lab-frame magnetic flux with, 
```py
eq = GFeq(**cases["circle"])
# Paramters with SI units to compute the magnetic flux over R, Z,
R0 = 1.0 # [m] 
B0 = 2.0 # [T]
T0 = 3 # [keV]
n0 = 6. # [E19 1/m^3]
p0 = n0*T0 * 1602.2 # [Pa]/[keV * E19/m^3]
# Main routine,
eq.get_PsiRZ(R0, B0, p0, almin=2.2, almax=2.4)
```
The ``almin, almax`` kwargs passed to ``get_PsiRZ`` help bound the search for the eigenvalue ``alpha``. 
__Warning__: This part of the program may require manual intervention!
In general we want to select the lowest eigenvalue. See the ``GFeq.get_alpha()`` method for more details.
The following plot is generated by the ``GFeq.get_alpha()`` method,

<p align="center">
  <img 
    width=“500”
    alt="image"
    src="https://github.com/user-attachments/assets/0347fc25-6077-43dc-9d3e-02cbb7e48d42"
  >
</p>

Several other results plots are generated.
For example, this figure shows the magnetic equilibrium along with midplane profiles of the plasma pressure, toroidal current density. We also show the pressure profile against the normalized poloidal magnetic flux (``psi``) and the q-profile.

<p align="center">
  <img 
    width=“1000”
    alt="image"
    src="https://github.com/user-attachments/assets/343d6c54-fd1c-4bd3-bd5f-3073d137469f"
  >
</p>

__Note__: calculating the q-profile requries flux-surface tracing and can be computationally intensive. See the ``GFeq.get_qprofile()`` method for more details.

The result plot for the "D" case is shown below, 
<p align="center">
  <img 
    width=“1000”
    alt="image"
    src="https://github.com/user-attachments/assets/f79436f3-b7ef-415b-8d52-25484dfdf9e4"
  >
</p>

## Benchmarking
The following table provides a comparison between values produced by this code and those published in Table 4 of [1].
Interestingly we do not completely agree as to the values of alpha. However, using the ``GFeq.check_GS()`` method I have verified that each value of ``alpha`` produced by this program results in a maximum |LHS - RHS| GS error on the order of 1E-15.

| case    | eps  | kappa | delta | nu  | alpha [1] | alpha (this code) |
|---------|------|-------|-------|-----|-----------|-------------------|
| circle  | 0.33 | 1     | 0     | 1   | 2.38      | 2.3577            |
| ellipse | 0.25 | 2     | 0     | 1   | 1.88      | 1.8724            |
| D       | 0.33 | 1.8   | 0.4   | 0.3 | 1.96      | 1.9057            |
| negD    | 0.33 | 1.9   | -0.6  | 0.5 | 1.91      | 1.8744            |

## References
[1] J. Plasma Phys. (2021), vol. 87, 905870303; https://doi.org/10.1017/S002237782100009X

[2] Physics of Plasmas 5, 973 (1998); https://doi.org/10.1063/1.872666
