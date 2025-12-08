import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv  # Modified Bessel function of the second kind
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

class TransferFunctions:
    """Collection of transfer functions equivalent to the MATLAB .m files"""
    
    @staticmethod
    def abos1(u, p):
        """Equivalent to abos1.m - Pore pressure"""
        K, alpha, ch, P, S, M, M11, M12, c2, thetar, cf, Rw, Pw, Po, G, r, theta, M13 = p
        
        F2 = (Pw - Po)
        Phy2 = kv(0, r * np.sqrt(u / c2)) / kv(0, Rw * np.sqrt(u / c2))
        
        A1 = (alpha * M) / (M11 + alpha**2 * M)
        A2 = (M11 + M12 + 2 * alpha**2 * M) / (M11 + alpha**2 * M)
        
        B1 = (M11 / (2 * G * alpha)) * kv(2, Rw * np.sqrt(u / c2))
        B2 = (1 / (np.sqrt(u / c2) * Rw)) * kv(1, Rw * np.sqrt(u / c2)) + \
             (6 / (np.sqrt(u / c2) * Rw)**2) * kv(2, Rw * np.sqrt(u / c2))
        B3 = 2 * ((1 / (np.sqrt(u / c2) * Rw)) * kv(1, Rw * np.sqrt(u / c2)) + \
                  (3 / (np.sqrt(u / c2) * Rw)**2) * kv(2, Rw * np.sqrt(u / c2)))
        
        C1 = 4 / (2 * A1 * (B3 - B2) - A2 * B1)
        C2 = -4 * B1 / (2 * A1 * (B3 - B2) - A2 * B1)
        C3 = (2 * A1 * (B2 + B3) + 3 * A2 * B1) / (3 * (2 * A1 * (B3 - B2) - A2 * B1))
        
        y = (Po + S * np.cos(2 * (theta - thetar)) * 
             ((cf / (2 * G * K)) * C1 * kv(2, r * np.sqrt(u / c2)) + 
              A1 * C2 * (Rw**2 / r**2))) / u
        
        return y
    
    @staticmethod
    def abos2(u, p):
        """Equivalent to abos2.m - Radial stress"""
        (alphaf, alpham, Vf, X, Taverage, K, alpha, phy, DT, ch, c1, P, S, M, M11, M12, c2, 
         Inc, InTemperature, thetar, cf, Rw, Pw, Po, G, R, r, theta, Betam, Gammam, M13) = p
        
        F2 = ((Pw - Po) - (c2 / K * ((alphaf - alpham) * phy - alpha * 
              (M11 * alpham + M12 * alpham + M13 * alpham) / M11) / (1 - c2 / ch) * InTemperature))
        F3 = ((c2 / K * ((alphaf - alpham) * phy - alpha * 
               (M11 * alpham + M12 * alpham + M13 * alpham) / M11) / (1 - c2 / ch)) * InTemperature)
        
        Phy1 = kv(0, r * np.sqrt(u / c1)) / kv(0, Rw * np.sqrt(u / c1))
        Phy2 = kv(0, r * np.sqrt(u / c2)) / kv(0, Rw * np.sqrt(u / c2))
        Phy3 = kv(0, r * np.sqrt(u / ch)) / kv(0, Rw * np.sqrt(u / ch))
        
        cha1 = (kv(1, r * np.sqrt(u / c1)) / (np.sqrt(u / c1) * r * kv(0, Rw * np.sqrt(u / c1)))) - \
               (kv(1, np.sqrt(u / c1) * Rw) * Rw / (np.sqrt(u / c1) * r**2 * kv(0, Rw * np.sqrt(u / c1))))
        cha2 = (kv(1, r * np.sqrt(u / c2)) / (np.sqrt(u / c2) * r * kv(0, Rw * np.sqrt(u / c2)))) - \
               (kv(1, np.sqrt(u / c2) * Rw) * Rw / (np.sqrt(u / c2) * r**2 * kv(0, Rw * np.sqrt(u / c2))))
        cha3 = (kv(1, r * np.sqrt(u / ch)) / (np.sqrt(u / ch) * r * kv(0, Rw * np.sqrt(u / ch)))) - \
               (kv(1, np.sqrt(u / ch) * Rw) * Rw / (np.sqrt(u / ch) * r**2 * kv(0, Rw * np.sqrt(u / ch))))
        
        A1 = (alpha * M) / (M11 + alpha**2 * M)
        A2 = (M11 + M12 + 2 * alpha**2 * M) / (M11 + alpha**2 * M)
        
        B1 = (M11 / (2 * G * alpha)) * kv(2, Rw * np.sqrt(u / c2))
        B2 = (1 / (np.sqrt(u / c2) * Rw)) * kv(1, Rw * np.sqrt(u / c2)) + \
             (6 / (np.sqrt(u / c2) * Rw)**2) * kv(2, Rw * np.sqrt(u / c2))
        B3 = 2 * ((1 / (np.sqrt(u / c2) * Rw)) * kv(1, Rw * np.sqrt(u / c2)) + \
                  (3 / (np.sqrt(u / c2) * Rw)**2) * kv(2, Rw * np.sqrt(u / c2)))
        
        C1 = 4 / (2 * A1 * (B3 - B2) - A2 * B1)
        C2 = -4 * B1 / (2 * A1 * (B3 - B2) - A2 * B1)
        C3 = (2 * A1 * (B2 + B3) + 3 * A2 * B1) / (3 * (2 * A1 * (B3 - B2) - A2 * B1))
        
        y = ((-P + S * np.cos(2 * (theta - thetar)) + (P - Pw) * (Rw**2 / r**2) + 
              S * (A1 * C1 * ((1 / (np.sqrt(u / c2) * r)) * kv(1, r * np.sqrt(u / c2)) + 
                               (6 / (np.sqrt(u / c2) * r)**2) * kv(2, r * np.sqrt(u / c2))) - 
                   A2 * C2 * (Rw**2 / r**2) - 3 * C3 * (Rw**4 / r**4)) * 
              np.cos(2 * (theta - thetar)))) / u
        
        return y
    
    @staticmethod
    def abos2t(u, p):
        """Equivalent to abos2t.m - Thermal component of radial stress"""
        (alphaf, alpham, Vf, X, Taverage, K, alpha, phy, DT, ch, c1, P, S, M, M11, M12, c2, 
         Inc, InTemperature, thetar, cf, Rw, Pw, Po, G, R, r, theta, Betam, Gammam, M13) = p
        
        cha3 = (kv(1, r * np.sqrt(u / ch)) / (np.sqrt(u / ch) * r * kv(0, Rw * np.sqrt(u / ch)))) - \
               (kv(1, np.sqrt(u / ch) * Rw) * Rw / (np.sqrt(u / ch) * r**2 * kv(0, Rw * np.sqrt(u / ch))))
        
        y = (Betam * (1 - M12 / M11) * (-(0.5002 / u**0.4524)) * cha3) / u
        
        return y
    
    @staticmethod
    def abos3(u, p):
        """Equivalent to abos3.m - Tangential stress"""
        (alphaf, alpham, Vf, X, Taverage, K, alpha, phy, DT, ch, c1, P, S, M, M11, M12, c2, 
         Inc, InTemperature, thetar, cf, Rw, Pw, Po, G, R, r, theta, Betam, Gammam, M13) = p
        
        A1 = (alpha * M) / (M11 + alpha**2 * M)
        A2 = (M11 + M12 + 2 * alpha**2 * M) / (M11 + alpha**2 * M)
        
        B1 = (M11 / (2 * G * alpha)) * kv(2, Rw * np.sqrt(u / c2))
        B2 = (1 / (np.sqrt(u / c2) * Rw)) * kv(1, Rw * np.sqrt(u / c2)) + \
             (6 / (np.sqrt(u / c2) * Rw)**2) * kv(2, Rw * np.sqrt(u / c2))
        B3 = 2 * ((1 / (np.sqrt(u / c2) * Rw)) * kv(1, Rw * np.sqrt(u / c2)) + \
                  (3 / (np.sqrt(u / c2) * Rw)**2) * kv(2, Rw * np.sqrt(u / c2)))
        
        C1 = 4 / (2 * A1 * (B3 - B2) - A2 * B1)
        C2 = -4 * B1 / (2 * A1 * (B3 - B2) - A2 * B1)
        C3 = (2 * A1 * (B2 + B3) + 3 * A2 * B1) / (3 * (2 * A1 * (B3 - B2) - A2 * B1))
        
        y = ((-P - S * np.cos(2 * (theta - thetar)) - (P - Pw) * (Rw**2 / r**2) + 
              S * (-A1 * C1 * ((1 / (np.sqrt(u / c2) * r)) * kv(1, r * np.sqrt(u / c2)) + 
                                (6 / (np.sqrt(u / c2) * r)**2 + 1) * kv(2, r * np.sqrt(u / c2))) + 
                   3 * C3 * (Rw**4 / r**4)) * np.cos(2 * (theta - thetar)))) / u
        
        return y
    
    @staticmethod
    def abos3t(u, p):
        """Equivalent to abos3t.m - Thermal component of tangential stress"""
        (alphaf, alpham, Vf, X, Taverage, K, alpha, phy, DT, ch, c1, P, S, M, M11, M12, c2, 
         Inc, InTemperature, thetar, cf, Rw, Pw, Po, G, R, r, theta, Betam, Gammam, M13) = p
        
        Phy3 = kv(0, r * np.sqrt(u / ch)) / kv(0, Rw * np.sqrt(u / ch))
        cha3 = (kv(1, r * np.sqrt(u / ch)) / (np.sqrt(u / ch) * r * kv(0, Rw * np.sqrt(u / ch)))) - \
               (kv(1, np.sqrt(u / ch) * Rw) * Rw / (np.sqrt(u / ch) * r**2 * kv(0, Rw * np.sqrt(u / ch))))
        
        y = (-Betam * (1 - M12 / M11) * (-(0.5002 / u**0.4524)) * (Phy3 + cha3)) / u
        
        return y
    
    @staticmethod
    def abos5(u, p):
        """Equivalent to abos5.m - Shear stress"""
        K, alpha, ch, P, S, M, M11, M12, c2, thetar, cf, Rw, Pw, Po, G, r, theta, M13 = p
        
        A1 = (alpha * M) / (M11 + alpha**2 * M)
        A2 = (M11 + M12 + 2 * alpha**2 * M) / (M11 + alpha**2 * M)
        
        B1 = (M11 / (2 * G * alpha)) * kv(2, Rw * np.sqrt(u / c2))
        B2 = (1 / (np.sqrt(u / c2) * Rw)) * kv(1, Rw * np.sqrt(u / c2)) + \
             (6 / (np.sqrt(u / c2) * Rw)**2) * kv(2, Rw * np.sqrt(u / c2))
        B3 = 2 * ((1 / (np.sqrt(u / c2) * Rw)) * kv(1, Rw * np.sqrt(u / c2)) + \
                  (3 / (np.sqrt(u / c2) * Rw)**2) * kv(2, Rw * np.sqrt(u / c2)))
        
        C1 = 4 / (2 * A1 * (B3 - B2) - A2 * B1)
        C2 = -4 * B1 / (2 * A1 * (B3 - B2) - A2 * B1)
        C3 = (2 * A1 * (B2 + B3) + 3 * A2 * B1) / (3 * (2 * A1 * (B3 - B2) - A2 * B1))
        
        y = ((-S * np.sin(2 * (theta - thetar)) + 
              S * (2 * A1 * C1 * ((1 / (np.sqrt(u / c2) * r)) * kv(1, r * np.sqrt(u / c2)) + 
                                   (3 / (np.sqrt(u / c2) * r)**2) * kv(2, r * np.sqrt(u / c2))) - 
                   A2 / 2 * C2 * (Rw**2 / r**2) - 3 * C3 * (Rw**4 / r**4)) * 
              np.sin(2 * (theta - thetar)))) / u
        
        return y
    
    @staticmethod
    def abost(u, p):
        """Equivalent to abost.m - Temperature"""
        (alphaf, alpham, Vf, X, Taverage, K, alpha, phy, DT, ch, c1, P, S, M, M11, M12, c2, 
         Inc, InTemperature, thetar, cf, Rw, Pw, Po, G, R, r, theta, M13) = p
        
        Phy3 = kv(0, r * np.sqrt(u / ch)) / kv(0, Rw * np.sqrt(u / ch))
        
        y = -(0.5002 / u**1.4524) * Phy3
        
        return y


def yan_inverse_laplace_transform(F_func, t, P=None):
    """
    Inverse Laplace transform using de Hoog, Knight, and Stokes algorithm
    """
    alpha = 0
    tol = 1e-9
    M = 10
    
    print('yan_inverse_laplace_transform: Work in progress')
    
    t = np.array(t, ndmin=1)
    allt = t
    logallt = np.log10(allt)
    iminlogallt = int(np.floor(np.min(logallt)))
    imaxlogallt = int(np.ceil(np.max(logallt)))
    
    f = []
    
    for ilogt in range(iminlogallt, imaxlogallt + 1):
        t_subset = allt[(logallt >= ilogt) & (logallt < (ilogt + 1))]
        
        if len(t_subset) > 0:
            T = np.max(t_subset) * 2
            gamma = alpha - np.log(tol) / (2 * T)
            
            nt = len(t_subset)
            run = np.arange(0, 2 * M + 1)
            s_vec = gamma + 1j * np.pi * run / T
            a = np.zeros(len(s_vec), dtype=complex)
            
            for k, s in enumerate(s_vec):
                print('.', end='')
                a[k] = F_func(s, P)
            
            print()
            a[0] = a[0] / 2
            
            # Continue with the algorithm
            e = np.zeros((2 * M + 1, M + 1), dtype=complex)
            q = np.zeros((2 * M, M + 1), dtype=complex)
            
            q[:, 1] = a[1:2*M+1] / a[0:2*M]
            
            for r in range(2, M + 2):
                e_range = 2 * (M - r + 1) + 1
                if e_range > 0:
                    e[:e_range, r-1] = (q[1:e_range+1, r-1] - q[:e_range, r-1] + 
                                       e[1:e_range+1, r-2])
                
                if r < M + 1:
                    q_range = 2 * (M - r) + 2
                    if q_range > 0:
                        e_vals = e[1:q_range+1, r-1]
                        e_vals_prev = e[:q_range, r-1]
                        # Avoid division by zero
                        nonzero_mask = np.abs(e_vals_prev) > 1e-15
                        q[:q_range, r] = 0
                        if np.any(nonzero_mask):
                            q[:q_range, r][nonzero_mask] = (q[1:q_range+1, r-1][nonzero_mask] * 
                                                           e_vals[nonzero_mask] / 
                                                           e_vals_prev[nonzero_mask])
            
            d = np.zeros(2 * M + 1, dtype=complex)
            d[0] = a[0]
            if M > 0:
                d[1:2*M:2] = -q[0, 1:M+1]
                d[2:2*M+1:2] = -e[0, 1:M+1]
            
            A = np.zeros((2 * M + 2, nt), dtype=complex)
            B = np.zeros((2 * M + 2, nt), dtype=complex)
            
            A[1, :] = d[0]
            B[0, :] = 1
            B[1, :] = 1
            
            z = np.exp(1j * np.pi * t_subset / T)
            
            for n in range(2, 2 * M + 2):
                if n - 1 < len(d):
                    A[n, :] = A[n-1, :] + d[n-1] * z * A[n-2, :]
                    B[n, :] = B[n-1, :] + d[n-1] * z * B[n-2, :]
            
            if 2 * M < len(d) and 2 * M + 1 < len(d):
                h2M = 0.5 * (1 + (d[2*M] - d[2*M+1]) * z)
                # Avoid division by zero
                h2M_safe = np.where(np.abs(h2M) < 1e-15, 1e-15, h2M)
                R2Mz = -h2M * (1 - np.sqrt(1 + d[2*M+1] * z / h2M_safe**2))
                A[2*M+1, :] = A[2*M, :] + R2Mz * A[2*M-1, :]
                B[2*M+1, :] = B[2*M, :] + R2Mz * B[2*M-1, :]
            
            # Avoid division by zero in final calculation
            B_final = B[2*M+1, :]
            B_safe = np.where(np.abs(B_final) < 1e-15, 1e-15, B_final)
            
            fpiece = (1 / T * np.exp(gamma * t_subset) * 
                     np.real(A[2*M+1, :] / B_safe))
            
            f.extend(fpiece)
    
    return np.array(f)


def create_polar_contour_plot(data, title="Polar Contour Plot"):
    """Create a polar contour plot similar to polarplot3d"""
    
    # Create polar coordinates
    n_angles, n_radii = data.shape
    angles = np.linspace(0, 2*np.pi, n_angles)
    radii = np.linspace(1, 2, n_radii)  # Normalized radial range
    
    # Create meshgrid
    R, THETA = np.meshgrid(radii, angles)
    
    # Convert to Cartesian coordinates
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Create contour plot
    contour = ax.contourf(THETA, R, data, levels=20, cmap='RdYlBu_r')
    
    # Add colorbar
    plt.colorbar(contour, ax=ax, shrink=0.8)
    
    ax.set_title(title, pad=20)
    ax.set_ylim(0, 2)
    
    return fig, ax


def main():
    """Main program equivalent to the MATLAB script"""
    
    # Initialize coordinate transformation matrices
    alphas = 0  # North-East direction
    betas = 0
    
    E = np.array([
        [np.cos(alphas)*np.cos(betas), np.sin(alphas)*np.cos(betas), np.sin(betas)],
        [-np.sin(alphas), np.cos(alphas), 0],
        [-np.cos(alphas)*np.sin(betas), -np.sin(alphas)*np.sin(betas), np.cos(betas)]
    ])
    
    # Stress components
    theH = 2.29 * 6800 * 0.00981 * 1e6  # Maximum horizontal stress
    theh = 1.76 * 6800 * 0.00981 * 1e6  # Minimum horizontal stress
    thev = 2.23 * 6800 * 0.00981 * 1e6  # Overburden stress
    
    ICS = np.array([[theH, 0, 0], [0, theh, 0], [0, 0, thev]])
    GCS = E.T @ ICS @ E
    
    # Wellbore coordinate transformation
    alphab = 0  # Well deviation
    betab = 0   # Azimuth
    
    B = np.array([
        [np.cos(alphab)*np.cos(betab), np.cos(alphab)*np.sin(betab), -np.sin(alphab)],
        [-np.sin(betab), np.cos(betab), 0],
        [np.sin(alphab)*np.cos(betab), np.sin(alphab)*np.sin(betab), np.cos(alphab)]
    ])
    
    BCS = B @ GCS @ B.T
    Sx, Sy, Sz = BCS[0,0], BCS[1,1], BCS[2,2]
    Sxy, Sxz, Syz = BCS[0,1], BCS[0,2], BCS[1,2]
    
    # Weak plane coordinate transformation
    alphaw = 66 * np.pi / 180  # Strike direction
    betaw = 55 * np.pi / 180   # Dip angle
    
    W = np.array([
        [np.cos(alphaw)*np.sin(betaw), np.sin(alphaw)*np.sin(betaw), np.cos(betaw)],
        [-np.sin(alphaw), np.cos(alphaw), 0],
        [-np.cos(alphaw)*np.cos(betaw), -np.sin(alphaw)*np.cos(betaw), np.sin(betaw)]
    ])
    
    Sw = 1.5e6
    PHYW = 18 * np.pi / 180
    
    # Initialize result arrays
    ABCD = np.ones((37, 21))
    
    # Material properties
    E_mod = 18370e6  # Elastic modulus (Pa)
    v = 0.189        # Poisson's ratio
    a = 1            # Anisotropy index for elastic modulus
    b = 1            # Anisotropy index for Poisson's ratio
    
    Epie = E_mod / a
    vpie = v / b
    M = 35800e6      # Biot modulus (Pa)
    Ks = 48200e6     # Matrix modulus (Pa)
    
    M11 = E_mod * (Epie - E_mod * vpie**2) / ((1 + v) * (Epie - Epie * v - 2 * E_mod * vpie**2))
    M12 = E_mod * (Epie * v + E_mod * vpie**2) / ((1 + v) * (Epie - Epie * v - 2 * E_mod * vpie**2))
    M13 = E_mod * Epie * vpie / (Epie - Epie * v - 2 * E_mod * vpie**2)
    M33 = Epie**2 * (1 - v) / (Epie - Epie * v - 2 * E_mod * vpie**2)
    
    alpha = 1 - (M11 + M12 + M13) / (3 * Ks)
    alphapie = 1 - (2 * M13 + M33) / (3 * Ks)
    G = E_mod / (2 * (1 + v))
    
    # Diffusion coefficients
    KK = 1e-7 * (1e-6)**2  # Permeability
    uu = 1e-3              # Viscosity
    K = KK / uu
    
    c2 = K * M * M11 / (alpha**2 * M + M11)
    ch = 0.01244 / 24 / 3600  # Thermal diffusivity
    cf = c2
    c1 = c2
    
    # Thermal parameters
    alphaf = 1e-3        # Fluid thermal expansion
    alpham = 3.71e-5     # Matrix thermal expansion
    InTemperature = 0    # Temperature difference
    c = 1                # Thermal expansion coefficient
    alphampie = alpham / c
    Betam = M11 * alpham + M12 * alpham + M13 * alphampie
    Betampie = 2 * M13 * alpham + M33 * alphampie
    phy = 0.04           # Porosity
    DT = -0.5184e-7      # Thermal permeability coefficient
    
    # Chemical parameters
    T0 = 300           # Initial reservoir temperature (K)
    R = 8.314          # Gas constant
    Ms0 = 3.6e-2       # Reservoir solute molar fraction
    Vf = 1.8e-5        # Fluid partial molar volume
    X = 0.9            # Membrane efficiency
    Taverage = 390     # Average reservoir temperature
    W0 = 3e6           # Chemical potential expansion coefficient
    d = 1              # Chemical expansion coefficient
    W0pie = W0 / d
    Msaverage = 1.8e-2
    Mfaverage = 1 - Msaverage
    Gammam = W0 / Msaverage * (1 - Msaverage / Mfaverage)
    Gammampie = W0pie / Msaverage * (1 - Msaverage / Mfaverage)
    vsol = 1.7e-5 * Ms0 + 1.8e-5 * (1 - Ms0)
    Inc = 0
    
    # Other parameters
    P = (Sx + Sy) / 2
    S = 0.5 * np.sqrt((Sx - Sy)**2 + 4 * Sxy**2)
    thetar = 0.5 * np.arctan(2 * Sxy / (Sx - Sy))
    Rw = 0.2159  # Wellbore radius (m)
    Pw = 1.8 * 6800 * 0.00981 * 1e6
    Po = 1.7 * 6800 * 0.00981 * 1e6
    
    t = 1000000    # Time
    tt = 3600 * 16 # Time for thermal calculation
    
    # Initialize transfer functions
    tf = TransferFunctions()
    
    print("Starting calculations...")
    
    for j in range(37):
        for i in range(21):
            r = 0.2159 + 0.01 * i
            theta = (10 * j + 90) * np.pi / 180  # Angle from Sx counter-clockwise
            
            print(f"Processing j={j+1}/37, i={i+1}/21", end='\r')
            
            # Parameter arrays for different functions
            params_basic = [K, alpha, ch, P, S, M, M11, M12, c2, thetar, cf, Rw, Pw, Po, G, r, theta, M13]
            params_full = [alphaf, alpham, Vf, X, Taverage, K, alpha, phy, DT, ch, c1, P, S, M, M11, M12, c2, 
                          Inc, InTemperature, thetar, cf, Rw, Pw, Po, G, R, r, theta, Betam, Gammam, M13]
            params_thermal = [alphaf, alpham, Vf, X, Taverage, K, alpha, phy, DT, ch, c1, P, S, M, M11, M12, c2, 
                             Inc, InTemperature, thetar, cf, Rw, Pw, Po, G, R, r, theta, M13]
            
            try:
                # Calculate stress components using inverse Laplace transform
                abost_val = yan_inverse_laplace_transform(tf.abost, [tt], params_thermal)
                abos1_val = yan_inverse_laplace_transform(tf.abos1, [t], params_basic)
                abos2_val = -yan_inverse_laplace_transform(tf.abos2, [t], params_full)
                abos2t_val = -yan_inverse_laplace_transform(tf.abos2t, [tt], params_full)
                abos3_val = -yan_inverse_laplace_transform(tf.abos3, [t], params_full)
                abos3t_val = -yan_inverse_laplace_transform(tf.abos3t, [tt], params_full)
                abos5_val = -yan_inverse_laplace_transform(tf.abos5, [t], params_basic)
                
                # Handle NaN values
                abos2t_val = np.nan_to_num(abos2t_val, nan=0.0)
                abos5_val = np.nan_to_num(abos5_val, nan=0.0)
                
                # Combine stress components
                abos22 = abos2_val[0] + abos2t_val[0] if len(abos2t_val) > 0 else abos2_val[0]
                abos33 = abos3_val[0] + abos3t_val[0] if len(abos3t_val) > 0 else abos3_val[0]
                abos44 = (Sz - vpie * (Sx + Sy)) + vpie * (abos2_val[0] + abos3_val[0]) + \
                         (Betampie - 2 * vpie * Betam) * abost_val[0] if len(abost_val) > 0 else 0
                
                # Transform to weak plane coordinates
                C = np.array([[np.cos(theta), np.sin(theta), 0],
                             [-np.sin(theta), np.cos(theta), 0],
                             [0, 0, 1]])
                
                Trr = abos22 - alpha * abos1_val[0]
                Ttt = abos33 - alpha * abos1_val[0]
                Tzz = abos44 - alphapie * abos1_val[0]
                Trt = abos5_val[0] if len(abos5_val) > 0 else 0
                
                CCS = np.array([[Trr, Trt, 0],
                               [Trt, Ttt, 0],
                               [0, 0, Tzz]])
                
                WCS = W @ B.T @ C.T @ CCS @ C @ B @ W.T
                
                # Normal stress on weak plane
                Txx = WCS[2, 2]
                
                # Shear stress on weak plane
                Tao = np.sqrt(WCS[2, 1]**2 + WCS[2, 0]**2)
                
                # Stability criterion
                yb = Sw + np.tan(PHYW) * Txx - Tao
                
                ABCD[j, i] = yb
                
            except Exception as e:
                print(f"\nError at j={j}, i={i}: {e}")
                ABCD[j, i] = 0
    
    print("\nCalculations completed!")
    
    # Apply stability criterion
    ABCD[ABCD > 0] = 0
    ABCD[ABCD < 0] = 1
    
    # Create polar contour plot
    fig, ax = create_polar_contour_plot(ABCD, "Wellbore Stability Analysis")
    
    plt.tight_layout()
    plt.show()
    
    return ABCD


if __name__ == "__main__":
    result = main()

    print("Analysis complete!") 
井壁稳定模拟
