#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 20:19:49 2024

@author: Quinn Pratt

Description
-----------
This program produces analytic solutions to the Grad–Shafranov equation 
describing tokamak equilibria using a method developed by Guazzotto and 
Freidberg in [1].

In this program we create up/down symmetric equilibria with model flux surfaces
based on the Miller geometry [2]. Flux surface shapes are characterized with 
scalars: eps = a/R0 (inverse aspect ratio), kappa (elongation), and delta 
(triangularity). The parameter \nu is related to the poloidal beta.

The 'free functions' -- plasma pressure, p, and ff' -- are restricted to  
quadratic functions of \psi. 

To compute the poloidal magnetic flux, \Psi(R, Z), three additional parameters
are required: 
    (1) the vacuum toroidal field (B0)  
    (2) the on-axis plasma pressure (p0)
    (3) the axial major radius (R0)

References
----------
[1] J. Plasma Phys. (2021), vol. 87, 905870303; https://doi.org/10.1017/S002237782100009X
[2] Physics of Plasmas 5, 973 (1998); https://doi.org/10.1063/1.872666

All rights reserved
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from math import sqrt, pi
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from skimage import measure

class GFeq(object):
    def __init__(self, eps=0.33, kappa=1, delta=0, nu=1):
        """
        Guazzotto-Freidberg plasma equilibrium.

        Parameters
        ----------
        eps : float
            Inverse aspect ratio, epsilon = a/R0
        kappa : float
            Elongation.
        delta : float
            Triangularity.
        nu : float
            Related to \beta_p.

        """
        # Input parameters,
        self.eps = eps
        self.kappa = kappa
        self.delta = delta
        self.nu = nu
        
        # Useful quantities, 
        self.eps_hat = 2*self.eps/(1 + self.eps**2) # eq. 2.9 of [1]
        self.delta_hat = np.arcsin(self.delta) # under eq. 3.4 of [1]
        self.xd = self.delta + 0.5*self.eps*(1 - self.delta**2) # under eq. 3.5 of [1]
        # Equation 3.8 of [1],
        self.Lam1 = (1 - self.eps)*(1 - self.delta_hat)**2/self.kappa**2
        self.Lam2 = (1 + self.eps)*(1 + self.delta_hat)**2/self.kappa**2
        self.Lam3 = self.kappa/( (1 - self.eps*self.delta)**2*(1 - self.delta**2) )
        
        # Physical constants,
        self.mu0 = 1.2566E-6 # vacuum perm. [m]*[kg]/[s]^2/[A]^2
        
        # Numerical controls, 
        self.M = 50 # number of terms in the C_n(x), S_n(x) expansions.
        
            
    def get_PsiRZ(self, R0, B0, p0, Nx=257,Ny=257,get_qprofile=True,alpha=None,show_plot=True, **kwargs):
        """
        Main function to compute the poloidal magnetic flux over an R, Z grid.

        Parameters
        ----------
        R0 : float
            Axial (major) radial location [m]
        B0 : float
            On-axis toroidal vacuum magnetic field [T]
        p0 : float
            On-axis pressure [Pa]
        Nx : int (optional)
            Size of the radial grid
        Ny : int (optional)
            Size of vertical grid.
        get_qprofile : bool (optional)
            Calculate the q-profile (slows down this routine)
        alpha : float (optional)
            GS eigenvalue, bypasses the get_alpha() routine.
        show_plot : bool (optional)
            Show a series of plots.
        **kwargs : passed to the get_alpha() method.

        Returns
        -------
        R : np.1darray
            Lab-frame R coordinate.
        Z : np.1darray
            Lab-frame Z coordinate.
        Psi : np.2darray ~ (Z, R)
            Poloidal magnetic flux
        """
        # Add attrs to the class, 
        self.R0 = R0
        self.B0 = B0
        self.p0 = p0
        # 1D normalized grids,
        x = np.linspace(-1, 1, Nx) # -1 <= x <= 1
        y = np.linspace(-self.kappa, self.kappa, Ny) # -kappa <= y <= kappa 
        xx, yy = np.meshgrid(x,y)
        
        # 1. Compute the GS eigenvalue (alpha)
        if alpha is None:
            alpha = self.get_alpha(**kwargs)
        # add the final alpha eigenvalue to the class, 
        self.alpha = alpha
        
        # (Optional) Verify that alpha is a solution to the GS equation with:
        # self.check_GS(xx,yy,alpha)
        
        # 2. Compute the 2D normalized psi(x,y) = Psi/Psi0
        psi_xy = self.get_psi(xx,yy,alpha)
        
        # 3. Compute dB and Psi0,
        # Toroidal beta on-axis Sec. 6 of [1] ([units of mu0] * [Pa] = [T^2])
        beta0 = 2*self.mu0*p0/B0**2 
        # Plasma diamagnetism (dB/B0), eq. 6.4 of [1]
        dB = B0 * 0.5*beta0*(1 + self.eps**2)*(1 - self.nu)/self.nu # [T]
        print(f"INFO: dB = {dB:.3e} [T]")
        # Magnetic flux on-axis, eq. 6.5 of [1]
        Psi0 = self.eps*B0*R0**2/alpha * sqrt(beta0/self.nu) # [m^2 * T]
        print(f"INFO: Psi0 = {Psi0:.3e} [m^2*T]")
        # Add each of these to the class,
        self.beta0 = beta0
        self.dB = dB
        self.Psi0 = Psi0

        # 4. Un-normalize to RZ,
        a = self.eps*R0 # [m]
        r = 1 + self.eps**2 + 2*self.eps*x
        R = R0*np.sqrt(r) # [m]
        Z = a*y # [m]
        
        Psi = psi_xy * Psi0 # [m^2*T]
        
        # 5. Additional calculations,
        I, q0 = self.get_plasma_params(x, y)
        
        if get_qprofile:
            q, q_psi = self.get_qprofile(x,y,show_plot=show_plot)
            q_rho = np.sqrt(1 - q_psi) # define rho s.t. rho = 0(core), 1(sep) 
        
        # Get all of these radial profiles,
        R, psi, rho, p, jphi, p_rho = self.get_profiles(x)
        
        if show_plot:
            norm = Normalize(0, vmax=1)
            cmap=plt.get_cmap("viridis")
            cmap.set_under(color='w')
            ctf_kw = dict(cmap=cmap,norm=norm,levels=10)
            # --- 
            fig, (ax0, axp0) = plt.subplots(1,2,num="GFeq.get_PsiRZ psi_xy")
            
            # Compute the model surface: xs(theta), ys(theta)
            Nt = 101
            theta = np.linspace(0, 2*pi, Nt)
            xs = np.cos(theta + self.delta_hat*np.sin(theta)) - \
                0.5*self.eps*(np.sin(theta + self.delta_hat*np.sin(theta)))**2
            ys = self.kappa*np.sin(theta)
            
            ax0.plot(xs, ys, 'r-')
            ax0.contourf(x,y,psi_xy,**ctf_kw)
            ax0.set_aspect("equal")
            ax0.set_xlabel("x")
            ax0.set_ylabel("y")
            # Plot the pressure and toroidal current density vs. x, 
            axp0.plot(x, p/max(p), 'b-', label="norm. p")
            axp0.plot(x, jphi/max(jphi),'r-',label=r"norm. $j_\phi$")
            axp0.legend()
            axp0.set_xlabel("x (midplane)")
            fig.tight_layout()

            # --- 
            fig, (ax1, axp1) = plt.subplots(1,2,num="GFeq.get_PsiRZ PsiRZ")

            # Compute the model surface, 
            Rs = R0 + a*np.cos(theta + self.delta_hat*np.sin(theta)) 
            Zs = a*self.kappa*np.sin(theta)
            
            ax1.plot(Rs, Zs, 'r-')
            ax1.contourf(R,Z,Psi/Psi0,**ctf_kw)
            ax1.set_aspect("equal")
            ax1.set_xlabel("R [m]")
            ax1.set_ylabel("Z [m]")
            # Plot the pressure and toroidal current density vs. R, 
            axp1.plot(R, p/max(p), 'b-', label="norm. p(R)")
            axp1.plot(R, jphi/max(jphi),'r-',label=r"norm. $j_\phi(R)$")
            axp1.legend()
            axp1.set_xlabel("R [m] (midplane)")
            fig.tight_layout()
            
            # --- 
            # Create another figure with the dimensioned values,
            fig = plt.figure(figsize=(11,5),num="GFeq.get_PsiRZ final")
            gs = fig.add_gridspec(2,3)
            ax0 = fig.add_subplot(gs[:,0])
            ax1 = fig.add_subplot(gs[0,1])
            ax2 = fig.add_subplot(gs[1,1])
            ax3 = fig.add_subplot(gs[0,2])
            ax4 = fig.add_subplot(gs[1,2])
            ax0.set_aspect("equal")
            ax0.plot(Rs, Zs, 'r-',lw=2)
            ax0.set_xlabel("R [m]")
            ax0.set_ylabel("Z [m]")
            ctf = ax0.contourf(R,Z,Psi/Psi0,**ctf_kw)
            fig.colorbar(ctf, ax=ax0,label=r"Norm. pol. mag. flux $\psi = \Psi/\Psi_0$")#,orientation='horizontal')
            # For the pressure, assume the value is given in SI units,
            ax1.plot(R, p, 'b-')
            ax1.set_ylabel("Plasma pressure [Pa]")
            ax1.set_xlabel("R [m] (midplane)")
            # For the Toroidal current density the SI units are [A]/[m]^2
            ax2.plot(R, jphi, 'r-')
            ax2.set_ylabel(r"Toroidal current density, $j_\phi$ [A/m$^2$]")
            ax2.set_xlabel("R [m] (midplane)")
            # Also show the pressure and q profile vs. psi,
            ax3.plot(rho, p_rho, 'b-')
            ax3.set_ylabel("Plasma pressure [Pa]")
            if hasattr(self, "q0"):
                ax4.plot(0,self.q0,'g P')
            if get_qprofile:
                ax4.plot(q_rho, q,'g-')
            ax4.set_ylabel(r"$q(\psi)$")
            for a in [ax3, ax4]:
                a.set_xlabel(r"$\rho = \sqrt{1 - \psi}$")
            fig.tight_layout()

        return R, Z, Psi
    
    def get_profiles(self, x):
        """
        Method to obtain (radial/psi) profiles of various plasma equilibrium quantities.

        Parameters
        ----------
        x : np.1darray
            Normalized radial coordiante, -1 <= x <= 1

        Returns
        -------
        R : np.1darray
            Major radius [m]
        psi : np.1darray
            normalized poloidal magnetic flux 1(core), 0(sep)
        rho : np.1darray
            ~sqrt(psi) with 0(core) 1(sep)
        p : np.1darray
            pressure over all 'R'
        jphi : np.1darray
            toroidal current density [A/m^2] vs. R
        p_rho : np.1darray
            pressure over 'rho'.
            
        """
        # Compute R from x, 
        r = 1 + self.eps**2 + 2*self.eps*x
        R = R0*np.sqrt(r) # [m]
        
        # 5.3 Profiles,
        psi = self.get_psi(x, 0, self.alpha)
        # analytic psi derivative of F2,
        dF2_dpsi = (self.R0*self.B0)**2 * (4*self.dB/self.B0 * (psi/self.Psi0) )
        # Compute the plasma pressure -- eq. 2.2 of [1]
        p = self.p0*(psi)**2
        dp_dpsi = 2*self.p0*(psi/self.Psi0) # analytic deriv.
        # Compute the toroidal current density,
        jphi = R*dp_dpsi + 0.5/(R*self.mu0)*dF2_dpsi # [A/m^2]
        
        core_ind = np.argmin(abs(psi - 1.))
        # psi goes from 1(core) --> 0(sep). Reverse to define 'rho',
        rho = np.sqrt(1 - psi[core_ind:]) # from 0(core)-->1(sep)
        p_rho = p[core_ind:]
        
        return R, psi, rho, p, jphi, p_rho
    
    def get_plasma_params(self, x, y):
        """ Method to evaluate various plasma parameters from Sec. 6 of [1]
        """
        xx, yy = np.meshgrid(x,y)
        psi_xy = self.get_psi(xx, yy, self.alpha)
        
        # Toroidal plasma current, I, eq. 6.8 of [1]
        integrand = (1 + self.nu*self.eps_hat*xx)/(1 + self.eps_hat*xx)*psi_xy
        integral = np.trapz(np.trapz(integrand,y,axis=0), x)
        pre = self.eps*self.B0*self.R0*self.alpha*sqrt(self.beta0/self.nu)
        I = pre*integral/self.mu0
        print(f"INFO: Toroidal plasma current, I = {I/1E6:.3f} [MA]")
        
        # On-axis saftey factor,
        psi_mid = self.get_psi(x,0,self.alpha)
        psi1_ind = np.argmin(abs(psi_mid - 1.0)) # core index.
        psi_xx, psi_yy = self.get_psipp(x,0,self.alpha)
        # Evaluate values at psi=1
        F1 = self.F(1)
        psi_xx_psi1 = psi_xx[psi1_ind]
        psi_yy_psi1 = psi_yy[psi1_ind]
        x_psi1 = x[psi1_ind]
        r_psi1 = 1 + self.eps**2 + 2*self.eps*x_psi1
        q0 = F1/(R0*B0)*sqrt(self.nu/self.beta0) * \
            self.eps*self.alpha/(r_psi1*sqrt(psi_xx_psi1*psi_yy_psi1))
        print(f"INFO: On-axis q0 = {q0:.3f}")
        # Add class attrs, 
        self.I = I
        self.q0 = q0
        
        return I, q0
  
    def F(self, psi):
        """
        Method to evaluate the 'free function', F(psi) = RBphi
        cf. eq. 2.2 of [1],
        
        Parameters
        ----------
        psi : normalized poloidal flux, psi = Psi/Psi0
            DESCRIPTION.

        """
        return self.R0*self.B0*np.sqrt(1 + 2*self.dB/self.B0*psi**2)

    
    def get_qprofile(self,x,y,psi_min=0.05,show_plot=True):
        """
        Method to evaluate the q(psi) profile from psi_min to 1-psi_min.
        This method can be time consuming because it involves tracing flux surfaces.
        
        Parameters
        ----------
        psi_min : float, optional
            Minimum value of psi_min for flux surface tracing. 
            The default is 0.05.

        Returns
        -------
        q : np.1darray
            Saftey factor
        psi_val : np.1darray
            Psi values matching the q profile.

        """
        xx, yy = np.meshgrid(x, y)
        psi_xy = self.get_psi(xx, yy, self.alpha)
        # Computing the q-profile requires parameterized of the flux surfaces.
        # recall: psi = 0 is the separatrix.
        psi_vals = np.arange(psi_min, 1, psi_min)
        q = np.zeros(len(psi_vals))
        # Functions needed for measure.find_contours()
        fx = interp1d(np.arange(0,len(x)), x)
        fy = interp1d(np.arange(0,len(y)), y)            
        if show_plot:
            fig, (ax0, ax1) = plt.subplots(1,2,num="GFeq.get_qprofile")
            norm = Normalize(0,1,clip=True)
            cmap = plt.get_cmap('viridis')
            cmap.set_under("w")
            colors = cmap(np.linspace(0,1,len(psi_vals)))
            cnt = ax0.contour(x,y,psi_xy,levels=psi_vals,
                              norm=norm,cmap=cmap,zorder=1)
            fig.colorbar(cnt,ax=ax0,label=r"$\psi$")
            ax0.set_aspect("equal")
            ax0.set_xlabel("x")
            ax0.set_ylabel("y")
            ax0.plot([],[],'k-',label="plt.contour")
            ax0.plot([],[],'k--',label='measure.find_contours')
            ax0.legend()
        print("INFO: calculating q-profile...")
        for i, p in enumerate(psi_vals):
            print(f"* tracing flux surface for psi={p:.2f}")
            # Extract contours for these values,
            contours = measure.find_contours(psi_xy, p)
            # Interpolate indices to get xc, yc points...
            yc = fy(contours[0][:,0])
            xc = fx(contours[0][:,1])
            if show_plot:
                ax0.plot(xc, yc,'--',color=colors[i],zorder=2)
            ti = np.arctan2(yc,xc)
            srt_inds = np.argsort(ti)
            xc = xc[srt_inds]
            yc = yc[srt_inds]
            ti = ti[srt_inds]
            ti, uinds = np.unique(ti, return_index=True)
            xc = xc[uinds]
            yc = yc[uinds]
            print(f"* contour has {len(xc)} unique points")
            dxdt = np.gradient(xc, ti)
            dydt = np.gradient(yc, ti)
            # Get the /derivative/ of psi along each contour,
            psip_x, psip_y = self.get_psip(xc, yc, self.alpha)
            # Compute 'r' for all values of xc, 
            r =  1 + self.eps**2 + 2*self.eps*xc
            # Integrand in eq. 6.11 of [1]
            integrand = np.sqrt( (dxdt**2 + r*dydt**2)/(psip_y**2 + r*psip_x**2) )/r
            I = np.trapz(integrand, ti) # integrate over thetas
            q[i] = self.F(p)/(self.R0*self.B0)*sqrt(self.nu/self.beta0)*self.eps*self.alpha/(2*pi)*I
            print(f"* local q = {q[i]:.3f}")
            
        if show_plot:
            # flip psi so core = 0, sep = 1...
            ax1.plot(1-psi_vals, q,'g-')
            ax1.set_xlabel(r"Normalized $\psi_n = 1-\psi$") # core = 1, sep = 0.
            ax1.set_ylabel(r"$q(\psi)$")
            if hasattr(self, 'q0'):
                ax1.plot(0, self.q0,'g P')
            fig.tight_layout()
    
        return q, psi_vals
        
    
    def get_psi(self, x, y, alpha):
        """
        Normalized magnetic flux over normalized spatial grids x, y.
        Note: 0 <= psi(x,y) <= 1 on the interior of the solution domain.
        And, psi(xs, ys) = 0.

        Parameters
        ----------
        x : np.1darray
            Normalized radial coordiate, -1 <= x <= 1
        y : np.1darray
            Normalized vertical coordiante, -kappa <= y <= kappa

        Returns
        -------
        psi_xy : np.2darray
            Normalized magnetic flux with shape [Ny, Nx]
        """
        # Set up and solve the linear system: Au = b
        a = np.atleast_1d(alpha)
        A, b = self.Ab(a) # A : (6,6,1), b : (6,1)
        u = solve(A[:,:,0],b[:,0]) # u = [c2, s2, c3, s3, c4, s4]
        # Equation 3.2 of [1],
        psi_xy = np.cos(self.hn(1,alpha)*y)*self.C(x, 1, alpha)
        psi_00 = self.C(0,1,alpha)
        for i, n in zip([0,2,4],[2,3,4]):
            psi_xy += np.cos(self.hn(n,alpha)*y)*( u[i]*self.C(x,n,alpha) + \
                                                u[i+1]*self.S(x,n,alpha) )
            psi_00 += u[i]*self.C(0,n,alpha) + u[i+1]*self.S(0,n,alpha)
        
        # Re-scale the eigenfunction to have psi(0,0) = 1,
        return psi_xy/psi_00
    
    def get_psip(self, x, y, alpha):
        """
        First derivatives of the normalized poloidal magnetic flux.

        Parameters
        ----------
        x : np.1darray
            Normalized radial coordiate, -1 <= x <= 1
        y : np.1darray
            Normalized vertical coordiante, -kappa <= y <= kappa

        Returns
        -------
        psi_x : np.2darray
            d\psi/dx over x,y
        psi_y :
            d\psi/dy over x,y
        """
        # precompute the expansion coefficients,
        a = np.atleast_1d(alpha)
        A, b = self.Ab(a) # A : (6,6,1), b : (6,1)
        u = solve(A[:,:,0],b[:,0]) # u = [c2, s2, c3, s3, c4, s4]

        h1 = self.hn(1,alpha)
        psi_x = np.cos(h1*y)*self.Cp(x, 1, alpha)
        psi_y = -h1*np.sin(h1*y)*self.C(x, 1, alpha)
        psi_00 = self.C(0,1,alpha) # value of psi(x,y) on-axis for renormalization
        for i, n in zip([0,2,4],[2,3,4]):
            hn = self.hn(n,alpha)
            psi_x += np.cos(hn*y)*( u[i]*self.Cp(x,n,alpha) + \
                                                u[i+1]*self.Sp(x,n,alpha) )
            psi_y -= hn*np.sin(hn*y)*( u[i]*self.C(x,n,alpha) + \
                                                u[i+1]*self.S(x,n,alpha) )
            psi_00 += u[i]*self.C(0,n,alpha) + u[i+1]*self.S(0,n,alpha)
        return psi_x/psi_00, psi_y/psi_00
    
    def get_psipp(self, x, y, alpha):
        """
        Second derivatives of the normalized poloidal magnetic flux.

        Parameters
        ----------
        x : np.1darray
            Normalized radial coordiate, -1 <= x <= 1
        y : np.1darray
            Normalized vertical coordiante, -kappa <= y <= kappa

        Returns
        -------
        psi_xx : np.2darray
            d^2\psi/dx^2 over x,y
        psi_yy :
            d^2\psi/dy^2 over x,y
        """
        # precompute the expansion coefficients,
        a = np.atleast_1d(alpha)
        A, b = self.Ab(a) # A : (6,6,1), b : (6,1)
        u = solve(A[:,:,0],b[:,0]) # u = [c2, s2, c3, s3, c4, s4]
        # grid,
        h1 = self.hn(1,alpha)
        psi_xx = np.cos(h1*y)*self.Cpp(x, 1, alpha)
        psi_yy = -h1**2*np.cos(h1*y)*self.C(x, 1, alpha)
        psi_00 = self.C(0,1,alpha) # value of psi(x,y) on-axis for renormalization
        for i, n in zip([0,2,4],[2,3,4]):
            hn = self.hn(n,alpha)
            psi_xx += np.cos(hn*y)*( u[i]*self.Cpp(x,n,alpha) + \
                                                u[i+1]*self.Spp(x,n,alpha) )
            psi_yy -= hn**2*np.cos(hn*y)*( u[i]*self.C(x,n,alpha) + \
                                                u[i+1]*self.S(x,n,alpha) )
            psi_00 += u[i]*self.C(0,n,alpha) + u[i+1]*self.S(0,n,alpha)
        return psi_xx/psi_00, psi_yy/psi_00

    def check_GS(self,x,y,alpha):
        """
        Function to check if a given value of alpha satisfies the GS equation.
        cf. eq. 2.9 of [1]

        Parameters
        ----------
        x : np.1darray
            Normalized radial coord. (-1 <= x <= 1)
        y : np.1darray
            Normalized vertical coord. (-kappa <= y <= kappa)
        alpha : float
            Eigenvalue of the GS equation.

        Returns
        -------
        err : np.2darray
            Absolute value difference, |LHS-RHS|, of the GS eq.
        """
        # 1. Get noramlized psi(x,y)
        psi= self.get_psi(x,y,alpha)
        # 2. Get second derivatives w.r.t. x and y,
        psi_xx, psi_yy = self.get_psipp(x, y, alpha)
        # Eq. 2.9 of [1]
        LHS = (1 + self.eps_hat*x)*psi_xx + 1/(1 + self.eps**2)*psi_yy
        RHS = -alpha**2*(1 + self.eps_hat*self.nu*x)*psi
        
        err = abs(LHS - RHS)
        print(f"Maximum |LHS-RHS| GS Error = {np.amax(err):.2e}")
        return err            
    
    def get_alpha(self,almin=2.2,almax=2.4,N_coarse=50,show_plot=True,do_fine=True):
        """
        Method to solve for the eigenvalue \alpha
        Only positive eigenvalues need to be considered.
        The value of alpha determines the unknown magnetic flux on-axis.

        Returns
        -------
        alpha : float
            Eigenvalue of the GS equation.

        """
        
        alphas = np.linspace(almin, almax, N_coarse)
        err = self.E(alphas)
        
        # Find the 'lowest alpha corresponding to a minimum in E(alpha)'
        mi = np.argmin(err)
        min_alpha = alphas[mi]
        print(f"INFO: initial alpha search: alpha={min_alpha:.4f} with E(alpha)={err[mi]:.2e}")
        # Run a minimizer near this value,
        if do_fine:
            result = minimize(self.E, min_alpha)
            alpha = result.x[0]
            min_err = result.fun
            print(f"INFO: alpha minimize: alpha={alpha:.4f} with E(alpha)={min_err:.2e}")
        else:
            alpha = min_alpha
            min_err = err[mi]
        
        if show_plot:
            fig, ax = plt.subplots(1,1,num="GFeq.get_alpha")
            ax.plot(alphas,err, '-o')
            ax.axhline(0, color='k',ls='--')
            ax.plot(min_alpha,err[mi],'b P') # 1st pass.
            ax.plot(alpha, min_err,'r P') # Minimizer result.
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$E(\alpha)$")
        
        return alpha
    
    def E(self, alpha):
        """
        Method corresponding to the minimization target, E(\alpha) = 0
        cf. eq. 3.11 of [1]

        Returns
        -------
        E(\alpha) : np.1darray
            Error defined by eq. 3.11 of [1]

        """
        Na = len(alpha)
        # 1. Solve the linear system in eq. 3.9 of [1],
        A, b = self.Ab(alpha) # A : (6,6,Na), b : (6,Na)
        u = np.zeros((6,Na))
        for i in range(Na):
            u[:,i] = solve(A[:,:,i],b[:,i]) #  u = [c2, s2, c3, s3, c4, s4]
            
        u = np.concatenate((np.ones((1,Na)), u),axis=0) # append c1; u : (7,Na)
        # 2. Build the Q vector, 
        Q = np.zeros((7,Na))
        # the constant term is a special case,
        h1 = self.hn(1,alpha)
        t1 = np.cos(h1*self.kappa)*self.Cpp(-self.xd,1,alpha)
        t2 = self.Lam3*h1*np.sin(h1*self.kappa)*self.C(-self.xd,1,alpha)
        Q[0,:] = t1 - t2
        for ci, n in zip([1,3,5],[2,3,4]):
            si = ci+1 # sine-term index,
            hn = self.hn(n, alpha)
            hnk = hn*self.kappa
            Q[ci,:] = self.Lam3*hn*np.sin(hnk)*self.C(-self.xd,n,alpha) + \
                np.cos(hnk)*self.Cpp(-self.xd,n,alpha)
            Q[si,:] = self.Lam3*hn*np.sin(hnk)*self.S(-self.xd,n,alpha) + \
                np.cos(hnk)*self.Spp(-self.xd,n,alpha)
        # Compute E,
        #   dot Q : (7, Na) into each u : (7, Na) 
        out = ([np.dot(x,y) for x,y in zip(Q.T,u.T)]/np.linalg.norm(Q*u,ord=1,axis=0))**2
        
        return out
    
    def Ab(self, alpha):
        """
        Function defining the linear system A*u = b
        The vector u = [c2, s2, c3, s3, c4, s4].
        The vector b are those terms proportional to c1(:=1)

        Returns
        -------
        A : np.3darray : (6, 6, len(alpha))
            square matrix for each value of alpha defining the linear system.
        b : np.2darray : (6, len(alpha))
            RHS of the linear system: A*u = b
        """
        Na = len(alpha)
        # Pre-compute,
        h1 = self.hn(1,alpha)
        # 1. The b vector,
        b = [self.C(-1,1,alpha), # (a)
                self.C(1,1,alpha), # (b)
                np.cos(h1*self.kappa)*self.C(-self.xd,1,alpha), # (c)
                np.cos(h1*self.kappa)*self.Cp(-self.xd,1,alpha), # (d)
                self.Lam1*self.Cp(-1,1,alpha) - h1**2*self.C(-1,1,alpha), # (e)
                -self.Lam2*self.Cp(1,1,alpha) - h1**2*self.C(1,1,alpha), # (f)
                ]
        b = -1*np.array(b) # (6, Na), multiply by -1 to move to RHS.
        # 2. Build the system of equations,
        A = np.zeros((6,6,Na))
        # Fill the remaining terms,
        for ci, n in zip([0,2,4],[2,3,4]):
            si = ci+1 # sine-term index,
            # Now we go equation-by-equation, 
            # (a)
            A[0,ci,:] = self.C(-1,n,alpha)
            A[0,si,:] = self.S(-1,n,alpha)
            # (b)
            A[1,ci,:] = self.C(1,n,alpha)
            A[1,si,:] = self.S(1,n,alpha)
            # (c)
            chk = np.cos(self.hn(n,alpha)*self.kappa)
            A[2,ci,:] = chk*self.C(-self.xd,n,alpha)
            A[2,si,:] = chk*self.S(-self.xd,n,alpha)
            # (d)
            A[3,ci,:] = chk*self.Cp(-self.xd,n,alpha)
            A[3,si,:] = chk*self.Sp(-self.xd,n,alpha)
            # (e)
            A[4,ci,:] = self.Lam1*self.Cp(-1,n,alpha) - self.hn(n,alpha)**2*self.C(-1, n, alpha)
            A[4,si,:] = self.Lam1*self.Sp(-1,n,alpha) - self.hn(n,alpha)**2*self.S(-1, n, alpha)
            # (f)
            A[5,ci,:] = -self.Lam2*self.Cp(1,n,alpha) - self.hn(n,alpha)**2*self.C(1, n, alpha)
            A[5,si,:] = -self.Lam2*self.Sp(1,n,alpha) - self.hn(n,alpha)**2*self.S(1, n, alpha)
                
        return A, b
        
    
    def plot_CS(self, x, n, alpha, axC=None,axS=None):
        """
        Method to plot the sine/cosine-like functions Cn(x), Sn(x).

        Parameters
        ----------
        x : np.array
            indep. var.
        n : int
            Order (one of 1, 2, 3, 4)
        alpha : float
            GS eigenvalue.

        """
        C = self.C(x, n, alpha)
        S = self.S(x, n, alpha)
        if axC is None or axS is None:
            fig, (axC, axS) = plt.subplots(2,1, sharex=True)
        else:
            fig = plt.gcf()
        axC.plot(x, C,label=fr"$C_{n}(x)$, $\alpha$ = {alpha}")
        axS.plot(x, S,label=fr"$S_{n}(x)$, $\alpha$ = {alpha}")
        return fig, (axC, axS)
    
    def hn(self, n, alpha):
        if n == 1:
            return sqrt(1 + self.eps**2)*alpha
        elif n == 2:
            return sqrt(35/36)*sqrt(1 + self.eps**2)*alpha
        elif n == 3:
            return sqrt(13/49)*sqrt(1 + self.eps**2)*alpha
        else:
            return 0
        
    def kn(self, n, alpha):
        if n == 1:
            return 0.
        elif n == 2:
            return (1/6)*alpha
        elif n == 3:
            return (6/7)*alpha
        else:
            return alpha
        
    def C(self, x, n, alpha):
        """ Function defining the cosine-like function C_n(x).
        C_n(x) is a power-series expansion of sines/cosines.
        """
        kn = self.kn(n,alpha)
        # precompute all the coeffs, 
        a, b = self.coeffs(n, alpha,a0=1,b0=0)
        
        out = 0.
        for m in range(self.M):
            out += a[m] * x**m * np.cos(kn*x)
            out += b[m] * x**m * np.sin(kn*x)
        
        return out
    
    def Cp(self, x, n, alpha):
        """ Function defining the d/dx C_n(x).
        i.e. the first derivative of the cosine-like function C_n(x).
        """
        kn = self.kn(n,alpha)
        # precompute all the coeffs, 
        a, b = self.coeffs(n, alpha,a0=1,b0=0)
        # Manually compute the m=0 case to avoid RuntimeWarnings,
        out = -kn*a[0]*np.sin(kn*x) + kn*b[0]*np.cos(kn*x)
        for m in range(1, self.M):
            out += (-kn*a[m]*x**m + m*b[m]*x**(m-1)) * np.sin(kn*x)
            out += (kn*b[m]*x**m + m*a[m]*x**(m-1)) * np.cos(kn*x)
        
        return out
    
    def Cpp(self, x, n, alpha):
        """ Function defining the d^2/dx^2 C_n(x).
        i.e. the second derivative of the cosine-like function C_n(x).
        """
        lam2 = self.eps_hat*alpha**2*self.nu 
        pre = -(self.kn(n,alpha)**2 + lam2*x)/(1 + self.eps_hat*x)
        return pre*self.C(x,n,alpha)
    
    def S(self, x, n, alpha):
        """ Function defining the sine-like function S_n(x).
        S_n(x) is a power-series expansion of sines/cosines.
        """
        kn = self.kn(n,alpha)
        a, b = self.coeffs(n, alpha,a0=0,b0=1)
        out = 0.
        for m in range(self.M):
            out += a[m] * x**m * np.cos(kn*x)
            out += b[m] * x**m * np.sin(kn*x)
        
        return out
    
    def Sp(self, x, n, alpha):
        """ Function defining the d/dx S_n(x).
        i.e. the first derivative of the sine-like function S_n(x).
        """
        kn = self.kn(n,alpha)
        a, b = self.coeffs(n, alpha,a0=0,b0=1)
        # Manually compute the m=0 case to avoid RuntimeWarnings,
        out = -kn*a[0]*np.sin(kn*x) + kn*b[0]*np.cos(kn*x)
        for m in range(1,self.M):
            out += (-kn*a[m]*x**m + m*b[m]*x**(m-1)) * np.sin(kn*x)
            out += (kn*b[m]*x**m + m*a[m]*x**(m-1)) * np.cos(kn*x)
        
        return out
    
    def Spp(self, x, n, alpha):
        """ Function defining the d^2/dx^2 S_n(x).
        i.e. the second derivative of the sine-like function C_n(x).
        """
        lam2 = self.eps_hat*alpha**2*self.nu 
        pre = -(self.kn(n,alpha)**2 + lam2*x)/(1 + self.eps_hat*x)
        return pre*self.S(x,n,alpha)
        
   
    def coeffs(self, n, alpha, a0=1,b0=0):
        """ Coefficients of the Cn(x), Sn(x) expansions.
        a0, b0 set the starting values, a1,b1,a2,b2 = 0 always.
        """
        lam2 = self.eps_hat*alpha**2*self.nu # eq. A1 of [1]
        kn = self.kn(n, alpha)
        Na = len(np.atleast_1d(alpha))
        a = np.zeros((self.M, Na))
        b = np.zeros((self.M, Na))
        a[0], b[0] = a0, b0
        for m in range(3,self.M):
            pre = -1/(m*(m-1))
            a[m,:] = pre*( self.eps_hat*(m-1)*(m-2)*a[m-1] + \
                        (lam2 - self.eps_hat*kn**2)*a[m-3] + \
                        2*(m-1)*kn*b[m-1] + \
                        2*self.eps_hat*(m-2)*kn*b[m-2] )
            b[m,:] = pre*( self.eps_hat*(m-1)*(m-2)*b[m-1] + \
                        (lam2 - self.eps_hat*kn**2)*b[m-3] + \
                        -2*(m-1)*kn*a[m-1] + \
                        -2*self.eps_hat*(m-2)*kn*a[m-2] )
        
        return a, b
            
# Standard cases,   
cases = {"circle":dict(eps=0.33,kappa=1,delta=0,nu=1), # alpha = 2.3577
         "ellipse":dict(eps=0.25,kappa=2,delta=0,nu=1), # alpha = 1.8724
         "D":dict(eps=0.33, kappa=1.8, delta=0.4, nu=0.3), # alpha = 1.9057
         "negD":dict(eps=0.33, kappa=1.9, delta=-0.6, nu=0.5), # alpha = 1.8744 
         }

# Create an instance of the GF equilibrium.
eq = GFeq(**cases["D"])
# Paramters with SI units to compute the magnetic flux over R, Z,
R0 = 1.0 # [m] 
B0 = 2.0 # [T]
# on-axis pressure,
T0 = 3 # [keV]
n0 = 6. # [E19 1/m^3]
p0 = n0*T0 * 1602.2 # [Pa]/[keV * E19/m^3]

# %% Main routine,
# Note: bypass the determination of alpha by providing alpha as kwarg.
#       otherwise set almin, almax kwargs to bound the get_alpha method. 
eq.get_PsiRZ(R0, B0, p0, almin=1.8, almax=2.0)

# %% Analysis of profiles,
# Create a normalized radial array,
x = np.linspace(-1,1,257)
# Run the get_profiles routine,
R, psi, rho, p, jphi, p_rho = eq.get_profiles(x)
# Convert the pressure profile to [keV * E19/m^3]
p = p_rho/1602.2
# Plot, 
fig, ax = plt.subplots(1,1,num="Profiles")
ax.plot(rho, p, label="pressure")
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"p, n [E19 m$^{-3}$], T [keV]")
# Deconvolve the density and temperature profiles,
n = sqrt(n0/T0)*np.sqrt(p)
T = sqrt(T0/n0)*np.sqrt(p)
ax.plot(rho, n,label="density")
ax.plot(rho, T,label="temperature")
# verify, 
ax.legend()
