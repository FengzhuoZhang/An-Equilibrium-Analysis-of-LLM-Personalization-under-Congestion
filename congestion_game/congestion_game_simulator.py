import numpy as np
import math
from scipy.optimize import fsolve, brentq, minimize_scalar
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict
from abc import ABC, abstractmethod
import sys
sys.path.append('/Users/fengzz/Desktop/LLM_alignment_econ/rebuttal_code/congestion_game')
from debug_utils import setup_debugpy

@dataclass
class UserType:
    """Parameters for a single user type"""
    name: str
    params: Dict  # Store any parameters needed for error functions
    # def __post_init__(self):
    #     self.params["zeta"] = self.params["tau"] + 2 * self.params["sigma_bar"]**2 / self.params["pi"] + self.params["m"]**2

    def update_zeta(self):
        self.params["zeta"] = self.params["tau"] + 2 * self.params["sigma_bar"]**2 / self.params["pi"] + self.params["m"]**2
    def update_m(self):
        self.params["m"] = math.sqrt(self.params["zeta"] - self.params["tau"] - 2 * self.params["sigma_bar"]**2 / self.params["pi"])
    
class SystemParams:
    """System-wide parameters"""
    def __init__(self, R_sft: float, R_icl: float, p: float, q: float):
        self.R_sft = R_sft
        self.R_icl = R_icl
        self.p = p
        self.q = q

class ErrorFunction(ABC):
    """Abstract base class for error functions"""
    
    @abstractmethod
    def E_sft(self, t: UserType, N: float) -> float:
        """SFT error as a function of number of samples"""
        pass
    
    @abstractmethod
    def E_icl(self, t: UserType, N: float) -> float:
        """ICL error as a function of number of samples"""
        pass

class CongestionFunction(ABC):
    """Abstract base class for congestion function"""
    
    @abstractmethod
    def h(self, R: float) -> float:
        """Congestion cost as a function of resource level"""
        pass

    @abstractmethod
    def R_for_delta(self, delta: float) -> float:
        """R such that h(R) = delta, where delta = H - p (inverse of h)."""
        pass
    
    def h_bar(self, R: float, p: float) -> float:
        """Total cost per unit resource"""
        return p + self.h(R)

class GeneralMFECalculator:
    """
    General calculator for Mean-Field Equilibrium with custom error and congestion functions
    """
    
    def __init__(self, 
                 type1: UserType, 
                 type2: UserType, 
                 params: SystemParams,
                 error_fn: ErrorFunction,
                 congestion_fn: CongestionFunction):
        self.t1 = type1
        self.t2 = type2
        self.params = params
        self.error_fn = error_fn
        self.congestion_fn = congestion_fn
        
        # Store computed values
        self.H_sep_1 = None
        self.H_sep_2 = None
        self.N_sft_curves = {}
        self.N_icl_curves = {}
        
    def cost_sft(self, t: UserType, N: float, H: float) -> float:
        """Total cost for SFT with N samples at congestion level H"""
        if N <= 0:
            return float('inf')
        return self.error_fn.E_sft(t, N) + self.params.R_sft * N * H
    
    def cost_icl(self, t: UserType, N: float, H: float) -> float:
        """Total cost for ICL with N samples at congestion level H"""
        if N < 0:
            return float('inf')
        return self.error_fn.E_icl(t, N) + self.params.R_icl * N * H
    
    def N_sft_optimal(self, t: UserType, H: float, N_min: float = 1e-6, N_max: float = 1e6) -> float:
        """Find optimal number of SFT samples given H"""
        result = minimize_scalar(
            lambda N: self.cost_sft(t, N, H),
            bounds=(N_min, N_max),
            method='bounded'
        )
        return result.x if result.success else N_min
    
    def N_icl_optimal(self, t: UserType, H: float, N_min: float = 0.0, N_max: float = 1e6) -> float:
        """Find optimal number of ICL samples given H"""
        result = minimize_scalar(
            lambda N: self.cost_icl(t, N, H),
            bounds=(N_min, N_max),
            method='bounded'
        )
        return result.x if result.success else 0.0
    
    def compute_N_curves(self, t: UserType, H_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute N_sft and N_icl curves over a range of H values.
        Returns: (N_sft_array, N_icl_array)
        """
        N_sft_arr = np.zeros_like(H_range)
        N_icl_arr = np.zeros_like(H_range)
        
        for i, H in enumerate(H_range):
            N_sft_arr[i] = self.N_sft_optimal(t, H)
            N_icl_arr[i] = self.N_icl_optimal(t, H)
        
        return N_sft_arr, N_icl_arr
    
    def Phi_sft(self, t: UserType, H: float) -> float:
        """Minimal cost when using SFT at congestion level H"""
        N_opt = self.N_sft_optimal(t, H)
        return self.cost_sft(t, N_opt, H)
    
    def Phi_icl(self, t: UserType, H: float) -> float:
        """Minimal cost when using ICL at congestion level H"""
        N_opt = self.N_icl_optimal(t, H)
        return self.cost_icl(t, N_opt, H)
    
    def psi(self, t: UserType, H: float) -> float:
        """Difference between SFT and ICL minimal costs"""
        return self.Phi_sft(t, H) - self.Phi_icl(t, H)
    
    def compute_H_sep(self, t: UserType, H_min: float = 1e-3, H_max: float = 1e6) -> float:
        """
        Numerically compute H_sep where psi(H) = 0
        i.e., where SFT and ICL have equal minimal cost
        """
        # Sample the psi function to find bracketing interval
        H_samples = np.logspace(np.log10(H_min), np.log10(H_max), 200)
        psi_samples = [self.psi(t, H) for H in H_samples]
        
        # Find sign changes
        for i in range(len(psi_samples) - 1):
            if psi_samples[i] * psi_samples[i+1] < 0:
                try:
                    H_sep = brentq(lambda H: self.psi(t, H), H_samples[i], H_samples[i+1])
                    return H_sep
                except:
                    continue
        
        # If no sign change found, check which algorithm dominates
        if psi_samples[0] < 0:
            return H_min  # SFT always better
        else:
            return H_max  # ICL always better
    
    def e_function(self, H: float) -> float:
        """All users adopt SFT - aggregate resource consumption"""
        q = self.params.q
        N_sft_1 = self.N_sft_optimal(self.t1, H)
        N_sft_2 = self.N_sft_optimal(self.t2, H)
        return q * self.params.R_sft * N_sft_1 + (1 - q) * self.params.R_sft * N_sft_2
    
    def f_function(self, H: float) -> float:
        """Type 1 adopts SFT, Type 2 adopts ICL - aggregate resource consumption"""
        q = self.params.q
        N_sft_1 = self.N_sft_optimal(self.t1, H)
        N_icl_2 = self.N_icl_optimal(self.t2, H)
        return q * self.params.R_sft * N_sft_1 + (1 - q) * self.params.R_icl * N_icl_2
    
    def g_function(self, H: float) -> float:
        """All users adopt ICL - aggregate resource consumption"""
        q = self.params.q
        N_icl_1 = self.N_icl_optimal(self.t1, H)
        N_icl_2 = self.N_icl_optimal(self.t2, H)
        return q * self.params.R_icl * N_icl_1 + (1 - q) * self.params.R_icl * N_icl_2
    
    def find_fixed_point(self, agg_func: Callable[[float], float], 
                         H_min: float = None, H_max: float = 1e6) -> Optional[float]:
        """
        Find H* such that R(H*) = agg_func(H*) with H* = p + h(R*), i.e.
        R* = h^{-1}(H* - p) = agg_func(H*).
        """
        p = self.params.p
        if H_min is None:
            H_min = p + 1e-6
        
        def equation(H):
            if H <= p:
                return float('inf')
            R = self.congestion_fn.R_for_delta(H - p)
            return R - agg_func(H)
        
        try:
            # Sample to find bracketing interval
            H_test = np.logspace(np.log10(H_min), np.log10(H_max), 100)
            vals = [equation(h) for h in H_test]
            
            # Find sign changes
            for i in range(len(vals) - 1):
                if not np.isinf(vals[i]) and not np.isinf(vals[i+1]) and vals[i] * vals[i+1] < 0:
                    return brentq(equation, H_test[i], H_test[i+1])
            
            # Try fsolve as backup
            H_init = p + 100
            result = fsolve(equation, H_init, full_output=True)
            if result[2] == 1 and result[0][0] > p:
                H_candidate = result[0][0]
                # Verify it's actually a fixed point
                if abs(equation(H_candidate)) < 1e-4:
                    return H_candidate
            
            return None
        except Exception as e:
            print(f"Error in find_fixed_point: {e}")
            return None
    
    def calculate_equilibrium(self, verbose: bool = True) -> Dict:
        """
        Calculate equilibrium R* according to the general algorithm.
        Returns dict with equilibrium info.
        """
        p = self.params.p
        
        # Step 1: Compute H_sep for both types
        if verbose:
            print("Computing H_sep for both types...")
        
        self.H_sep_1 = self.compute_H_sep(self.t1)
        self.H_sep_2 = self.compute_H_sep(self.t2)
        
        if verbose:
            print(f"H_sep(type1) = {self.H_sep_1:.4f}")
            print(f"H_sep(type2) = {self.H_sep_2:.4f}")
        
        # Ensure H_sep_2 <= H_sep_1 by swapping if needed
        if self.H_sep_2 > self.H_sep_1:
            self.H_sep_1, self.H_sep_2 = self.H_sep_2, self.H_sep_1
            self.t1, self.t2 = self.t2, self.t1
            self.params.q = 1 - self.params.q
            if verbose:
                print("Swapped types to ensure H_sep_2 <= H_sep_1")
        
        # Step 2: Determine which case we're in and solve fixed point
        e_val = self.e_function(self.H_sep_2)
        f_val_2 = self.f_function(self.H_sep_2)
        f_val_1 = self.f_function(self.H_sep_1)
        g_val_1 = self.g_function(self.H_sep_1)
        
        case = None
        H_star = None
        
        # Case 1: H_sep_2 > p + h(e(H_sep_2))  (paper: h(R)=R^2 gives e^2 + p)
        if self.H_sep_2 > p + self.congestion_fn.h(e_val):
            case = 1
            if verbose:
                print("\nCase 1: All users adopt SFT")
            H_star = self.find_fixed_point(self.e_function)
        
        # Case 2: p + h(f(H_sep_2)) <= H_sep_2 <= p + h(e(H_sep_2))
        elif p + self.congestion_fn.h(f_val_2) <= self.H_sep_2 <= p + self.congestion_fn.h(e_val):
            case = 2
            if verbose:
                print("\nCase 2: Mixed at H_sep_2")
            H_star = self.H_sep_2
        
        # Case 3: H_sep_2 < p + h(f(H_sep_2)) and p + h(f(H_sep_1)) < H_sep_1
        elif self.H_sep_2 < p + self.congestion_fn.h(f_val_2) and p + self.congestion_fn.h(f_val_1) < self.H_sep_1:
            case = 3
            if verbose:
                print("\nCase 3: Type 1 SFT, Type 2 ICL")
            H_star = self.find_fixed_point(self.f_function)
        
        # Case 4: p + h(g(H_sep_1)) <= H_sep_1 <= p + h(f(H_sep_1))
        elif p + self.congestion_fn.h(g_val_1) <= self.H_sep_1 <= p + self.congestion_fn.h(f_val_1):
            case = 4
            if verbose:
                print("\nCase 4: Mixed at H_sep_1")
            H_star = self.H_sep_1
        
        # Case 5: H_sep_1 < p + h(g(H_sep_1))
        else:
            case = 5
            if verbose:
                print("\nCase 5: All users adopt ICL")
            H_star = self.find_fixed_point(self.g_function)
        
        # Handle case where fixed point wasn't found
        if H_star is None:
            if verbose:
                print(f"Warning: Could not find fixed point for case {case}, using p")
            H_star = p
        
        R_star = self.congestion_fn.R_for_delta(max(0, H_star - p))
        
        if verbose:
            print(f"\nEquilibrium results:")
            print(f"  Case: {case}")
            print(f"  H* = {H_star:.4f}")
            print(f"  R* = {R_star:.4f}")
        
        # Determine what each type does at equilibrium
        psi_1 = self.psi(self.t1, H_star)
        psi_2 = self.psi(self.t2, H_star)
        
        type1_choice = "SFT" if psi_1 < 0 else ("ICL" if psi_1 > 0 else "Mixed")
        type2_choice = "SFT" if psi_2 < 0 else ("ICL" if psi_2 > 0 else "Mixed")
        
        if verbose:
            print(f"  Type 1 chooses: {type1_choice}")
            print(f"  Type 2 chooses: {type2_choice}")
        
        return {
            'R_star': R_star,
            'H_star': H_star,
            'case': case,
            'H_sep_1': self.H_sep_1,
            'H_sep_2': self.H_sep_2,
            'type1_choice': type1_choice,
            'type2_choice': type2_choice,
            'psi_1': psi_1,
            'psi_2': psi_2
        }
    
    def plot_analysis(self, H_range: np.ndarray = None, figsize: Tuple = (15, 200)):
        """
        Create comprehensive plots of the equilibrium analysis
        """
        if H_range is None:
            H_min = self.params.p + 1
            H_max = max(self.H_sep_1, self.H_sep_2) * 2
            H_range = np.linspace(H_min, H_max, 200)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: N_sft curves
        N_sft_1, _ = self.compute_N_curves(self.t1, H_range)
        N_sft_2, _ = self.compute_N_curves(self.t2, H_range)
        
        axes[0, 0].plot(H_range, N_sft_1, label='Type 1', linewidth=2)
        axes[0, 0].plot(H_range, N_sft_2, label='Type 2', linewidth=2)
        axes[0, 0].axvline(self.H_sep_1, color='red', linestyle='--', alpha=0.5, label='H_sep_1')
        axes[0, 0].axvline(self.H_sep_2, color='blue', linestyle='--', alpha=0.5, label='H_sep_2')
        axes[0, 0].set_xlabel('H')
        axes[0, 0].set_ylabel('N_sft')
        axes[0, 0].set_title('Optimal SFT Samples')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: N_icl curves
        _, N_icl_1 = self.compute_N_curves(self.t1, H_range)
        _, N_icl_2 = self.compute_N_curves(self.t2, H_range)
        
        axes[0, 1].plot(H_range, N_icl_1, label='Type 1', linewidth=2)
        axes[0, 1].plot(H_range, N_icl_2, label='Type 2', linewidth=2)
        axes[0, 1].axvline(self.H_sep_1, color='red', linestyle='--', alpha=0.5, label='H_sep_1')
        axes[0, 1].axvline(self.H_sep_2, color='blue', linestyle='--', alpha=0.5, label='H_sep_2')
        axes[0, 1].set_xlabel('H')
        axes[0, 1].set_ylabel('N_icl')
        axes[0, 1].set_title('Optimal ICL Samples')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: psi functions
        psi_1 = [self.psi(self.t1, H) for H in H_range]
        psi_2 = [self.psi(self.t2, H) for H in H_range]
        
        axes[0, 2].plot(H_range, psi_1, label='Type 1', linewidth=2)
        axes[0, 2].plot(H_range, psi_2, label='Type 2', linewidth=2)
        axes[0, 2].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[0, 2].axvline(self.H_sep_1, color='red', linestyle='--', alpha=0.5, label='H_sep_1')
        axes[0, 2].axvline(self.H_sep_2, color='blue', linestyle='--', alpha=0.5, label='H_sep_2')
        axes[0, 2].set_xlabel('H')
        axes[0, 2].set_ylabel('ψ(H)')
        axes[0, 2].set_title('Cost Difference (SFT - ICL)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Aggregate functions e, f, g
        e_vals = [self.e_function(H) for H in H_range]
        f_vals = [self.f_function(H) for H in H_range]
        g_vals = [self.g_function(H) for H in H_range]
        
        axes[1, 0].plot(H_range, e_vals, label='e(H): All SFT', linewidth=2)
        axes[1, 0].plot(H_range, f_vals, label='f(H): T1=SFT, T2=ICL', linewidth=2)
        axes[1, 0].plot(H_range, g_vals, label='g(H): All ICL', linewidth=2)
        axes[1, 0].axvline(self.H_sep_1, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(self.H_sep_2, color='blue', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('H')
        axes[1, 0].set_ylabel('Aggregate Resource')
        axes[1, 0].set_title('Aggregate Demand Functions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Fixed point visualization
        R_vals = np.array([self.congestion_fn.R_for_delta(H - self.params.p) for H in H_range])
        axes[1, 1].plot(H_range, R_vals, label=r"$h^{-1}(H-p)$", linewidth=2, color='black')
        axes[1, 1].plot(H_range, e_vals, label='e(H)', linewidth=2, alpha=0.7)
        axes[1, 1].plot(H_range, f_vals, label='f(H)', linewidth=2, alpha=0.7)
        axes[1, 1].plot(H_range, g_vals, label='g(H)', linewidth=2, alpha=0.7)
        
        # Mark equilibrium
        result = self.calculate_equilibrium(verbose=False)
        axes[1, 1].plot(result['H_star'], result['R_star'], 'ro', markersize=10, 
                       label=f"Equilibrium (Case {result['case']})", zorder=5)
        
        axes[1, 1].set_xlabel('H')
        axes[1, 1].set_ylabel('R or Aggregate')
        axes[1, 1].set_title('Fixed Point Equation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Summary text
        axes[1, 2].axis('off')
        summary_text = f"""
Equilibrium Summary:
━━━━━━━━━━━━━━━━━━━━
Case: {result['case']}
H* = {result['H_star']:.2f}
R* = {result['R_star']:.2f}

Separator Values:
H_sep(Type 1) = {result['H_sep_1']:.2f}
H_sep(Type 2) = {result['H_sep_2']:.2f}

At Equilibrium:
Type 1: {result['type1_choice']}
Type 2: {result['type2_choice']}

System Parameters:
p = {self.params.p}
q = {self.params.q:.2f}
R_sft = {self.params.R_sft}
R_icl = {self.params.R_icl}
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, 
                       verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        return fig


# Example: Define custom error and congestion functions
class SimplifiedErrorFunction(ErrorFunction):
    """Simplified error functions from equation (12) in the paper"""
    
    def E_sft(self, t: UserType, N: float) -> float:
        if N <= 0:
            return float('inf')
        d = t.params['d']
        sigma_e = t.params['sigma_e']
        return 2 * sigma_e**2 * d / N
    
    def E_icl(self, t: UserType, N: float) -> float:
        r = t.params['r']
        d = t.params['d']
        sigma_e = t.params['sigma_e']
        pi = t.params['pi']
        tau = t.params['tau']
        zeta = t.params['zeta']
        
        return 2 * sigma_e**2 * r /( N + 2 * sigma_e**2 / zeta) + (d - r) * tau

class LinearCongestion(CongestionFunction):
    """h(R) = R"""
    
    def h(self, R: float) -> float:
        if R >= 0:
            return float(R)
        else:
            return 0.0

    def R_for_delta(self, delta: float) -> float:
        if delta >= 0:
            return float(delta)
        else:
            return 0.0

class QuadraticCongestion(CongestionFunction):
    """h(R) = R^2"""
    
    def h(self, R: float) -> float:
        return R**2

    def R_for_delta(self, delta: float) -> float:
        return float(np.sqrt(max(0.0, delta)))


class ExponentialCongestion(CongestionFunction):
    """h(R) = exp(R)"""
    
    def h(self, R: float) -> float:
        return np.exp(R)

    def R_for_delta(self, delta: float) -> float:
        d = max(float(delta), 1e-300)
        return float(np.log(d))


# Example usage
if __name__ == "__main__":
    setup_debugpy(force=True)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman']
    plt.rcParams['font.size'] = 20
    font_size = 20
    legend_font_size = 15
    linewidth = 2
    color_1 = np.array([230, 145,  70]) / 255  # #E69146 amber (not pale)
    color_2 = np.array([122,  43,   0]) / 255  # #7A2B00 burnt orange

    
    # Create calculator with custom functions
    error_fn = SimplifiedErrorFunction()
    # congestion_fn = QuadraticCongestion()
    # congestion_fn = ExponentialCongestion()
    congestion_fn = LinearCongestion()

    # Define parameters for two types
    scale = 1.0
    d = 300 * scale
    r = 100 * scale
    sigma_bar = math.sqrt(1.0) * scale
    sigma_e = 1.0 * scale
    m = 2.0 * scale
    zeta = 8.0 * scale
    pi = 1.0 * scale
    tau = 3.0 * scale
    Rsft = 27.0
    Ricl = 4.5
    p = 10.0 * scale
    type1 = UserType(
        name="Type 1",
        params={
            'd': d,
            'r': r,
            'sigma_e': sigma_e,
            'sigma_bar': sigma_bar,
            'm': sigma_bar,
            'pi': pi,
            'tau': tau,
            'zeta': zeta
        }
    )
    
    type2 = UserType(
        name="Type 2",
        params={
            'd': d,
            'r': r,
            'sigma_e': sigma_e,
            'sigma_bar': sigma_bar,
            'm': sigma_bar,
            'pi': pi,
            'tau': tau,
            'zeta': zeta
        }
    )

    type1.update_m()
    type2.update_m()
    # System parameters
    params = SystemParams(
        R_sft=Rsft,
        R_icl=Ricl,
        p=p,
        q=0.2
    )
    
    # calculator = GeneralMFECalculator(type1, type2, params, error_fn, congestion_fn)
    
    # # Calculate equilibrium
    # result = calculator.calculate_equilibrium(verbose=True)

    # print("\n" + "="*50)
    # print("FINAL EQUILIBRIUM:")
    # print("="*50)
    # print(f"R* = {result['R_star']:.4f}")
    # print(f"H* = {result['H_star']:.4f}")
    # print(f"Case: {result['case']}")
    

    #### pi
    # num_p = 8
    # pi_list = list(np.linspace(0.1, 20.0, 200))
    # p_list = list(np.linspace(0, 35.0, num_p))
    # param1_list = pi_list
    # r_list = list(range(1, int(d)-1))
    # # r_list = list(range(1, 20))
    # r_list = [float(r) for r in r_list]
    # param1_list = [float(param1) for param1 in param1_list]
    # param2_list = p_list
    # param2_list = [float(param2) for param2 in param2_list]
    # R_star_arrays = []
    # count = 0
    # for param2 in param2_list:
    #     print(f"Processing {count} of {len(param2_list)}")
    #     count += 1
    #     params.p = param2
    #     R_star_list = []
    #     for param1 in param1_list:
    #         type1.params['pi'] = param1
    #         type2.params['pi'] = param1
    #         type1.update_zeta()
    #         type2.update_zeta()
    #         calculator = GeneralMFECalculator(type1, type2, params, error_fn, congestion_fn)
    #         result = calculator.calculate_equilibrium(verbose=False)
    #         R_star_list.append(result['R_star'])
    #     R_star_arrays.append(R_star_list)

    # plt.figure()
    # for i in range(num_p):
    #     t = i / (num_p - 1) if num_p > 1 else 0
    #     color = (1 - t) * color_1 + t * color_2
    #     plt.plot(param1_list, R_star_arrays[i], label=f'$p$={param2_list[i]:.0f}',linewidth=linewidth, color=color)
    # plt.legend(fontsize=legend_font_size, frameon=True, fancybox=True, ncol=3)
    # plt.xlabel(r'$\pi$', fontsize=font_size)
    # plt.xticks([0, 5, 10, 15, 20])
    # plt.yticks([2, 4, 6, 8, 10])
    # plt.xlim(0, 20)
    # plt.ylim(2, 10)
    # plt.ylabel(r'$R^*$', fontsize=font_size)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('output/homo_general_linear_pi.pdf', dpi=300, bbox_inches='tight')
    
    #### figure of r
    # type1 = UserType(
    #     name="Type 1",
    #     params={
    #         'd': d,
    #         'r': r,
    #         'sigma_e': sigma_e,
    #         'sigma_bar': sigma_bar,
    #         'm': sigma_bar,
    #         'pi': pi,
    #         'tau': tau,
    #         'zeta': zeta
    #     }
    # )
    # type2 = UserType(
    #     name="Type 2",
    #     params={
    #         'd': d,
    #         'r': r,
    #         'sigma_e': sigma_e,
    #         'sigma_bar': sigma_bar,
    #         'm': sigma_bar,
    #         'pi': pi,
    #         'tau': tau,
    #         'zeta': zeta
    #     }
    # )

    # type1.update_m()
    # type2.update_m()
    # # System parameters
    # params = SystemParams(
    #     R_sft=Rsft,
    #     R_icl=Ricl,
    #     p=p,
    #     q=0.2
    # )
    # num_p = 8
    # r_list = list(np.linspace(0.1, 150.0, 200))
    # p_list = list(np.linspace(0, 14.0, num_p))
    # param1_list = r_list
    # param1_list = [float(param1) for param1 in param1_list]
    # param2_list = p_list
    # param2_list = [float(param2) for param2 in param2_list]
    # R_star_arrays = []
    # count = 0
    # for param2 in param2_list:
    #     print(f"Processing {count} of {len(param2_list)}")
    #     count += 1
    #     params.p = param2
    #     R_star_list = []
    #     for param1 in param1_list:
    #         type1.params['r'] = param1
    #         type2.params['r'] = param1
    #         type1.update_zeta()
    #         type2.update_zeta()
    #         calculator = GeneralMFECalculator(type1, type2, params, error_fn, congestion_fn)
    #         result = calculator.calculate_equilibrium(verbose=False)
    #         R_star_list.append(result['R_star'])
    #     R_star_arrays.append(R_star_list)

    # plt.figure()
    # for i in range(num_p):
    #     t = i / (num_p - 1) if num_p > 1 else 0
    #     color = (1 - t) * color_1 + t * color_2
    #     plt.plot(param1_list, R_star_arrays[i], label=f'$p$={param2_list[i]:.0f}',linewidth=linewidth, color=color)
    # plt.legend(fontsize=legend_font_size, frameon=True, fancybox=True, ncol=3)
    # plt.xlabel(r'$r$', fontsize=font_size)
    # plt.xticks([0, 30, 60, 90, 120, 150])
    # plt.yticks([0, 3, 6, 9, 12, 15])
    # plt.xlim(0, 150)
    # plt.ylim(0, 15)
    # plt.ylabel(r'$R^*$', fontsize=font_size)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('output/homo_general_linear_r.pdf', dpi=300, bbox_inches='tight')
    
    #### figure of sigma_e
    # type1 = UserType(
    #     name="Type 1",
    #     params={
    #         'd': d,
    #         'r': r,
    #         'sigma_e': sigma_e,
    #         'sigma_bar': sigma_bar,
    #         'm': sigma_bar,
    #         'pi': pi,
    #         'tau': tau,
    #         'zeta': zeta
    #     }
    # )
    # type2 = UserType(
    #     name="Type 2",
    #     params={
    #         'd': d,
    #         'r': r,
    #         'sigma_e': sigma_e,
    #         'sigma_bar': sigma_bar,
    #         'm': sigma_bar,
    #         'pi': pi,
    #         'tau': tau,
    #         'zeta': zeta
    #     }
    # )

    # type1.update_m()
    # type2.update_m()
    # # System parameters
    # params = SystemParams(
    #     R_sft=Rsft,
    #     R_icl=Ricl,
    #     p=p,
    #     q=0.2
    # )
    # num_p = 8
    # sigma_e_list = list(np.linspace(0.1, 12.0, 200))
    # pi_list = list(np.linspace(0.1, 1.5, num_p))
    # param1_list = sigma_e_list
    # param1_list = [float(param1) for param1 in param1_list]
    # param2_list = pi_list
    # param2_list = [float(param2) for param2 in param2_list]
    # R_star_arrays = []
    # count = 0
    # for param2 in param2_list:
    #     print(f"Processing {count} of {len(param2_list)}")
    #     count += 1
    #     type1.params['pi'] = param2
    #     type2.params['pi'] = param2
    #     R_star_list = []
    #     for param1 in param1_list:
    #         type1.params['sigma_e'] = param1
    #         type2.params['sigma_e'] = param1
    #         type1.update_zeta()
    #         type2.update_zeta()
    #         calculator = GeneralMFECalculator(type1, type2, params, error_fn, congestion_fn)
    #         result = calculator.calculate_equilibrium(verbose=False)
    #         R_star_list.append(result['R_star'])
    #     R_star_arrays.append(R_star_list)

    # plt.figure()
    # for i in range(num_p):
    #     t = i / (num_p - 1) if num_p > 1 else 0
    #     color = (1 - t) * color_1 + t * color_2
    #     plt.plot(param1_list, R_star_arrays[i], label=f'$\pi$={param2_list[i]:.0f}',linewidth=linewidth, color=color)
    # plt.legend(fontsize=legend_font_size, frameon=True, fancybox=True, ncol=3)
    # plt.xlabel(r'$\tilde{\sigma}$', fontsize=font_size)
    # plt.xticks([0, 3, 6, 9, 12])
    # plt.yticks([0, 5, 10, 15, 20, 25])
    # plt.xlim(0, 12)
    # plt.ylim(0, 25)
    # plt.ylabel(r'$R^*$', fontsize=font_size)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('output/homo_general_linear_sigma_e.pdf', dpi=300, bbox_inches='tight')
    


    ### figure of R_SFT-R_ICL
    type1 = UserType(
        name="Type 1",
        params={
            'd': d,
            'r': r,
            'sigma_e': sigma_e,
            'sigma_bar': sigma_bar,
            'm': sigma_bar,
            'pi': pi,
            'tau': tau,
            'zeta': zeta
        }
    )
    type2 = UserType(
        name="Type 2",
        params={
            'd': d,
            'r': r,
            'sigma_e': sigma_e,
            'sigma_bar': sigma_bar,
            'm': sigma_bar,
            'pi': pi,
            'tau': tau,
            'zeta': zeta
        }
    )

    type1.update_m()
    type2.update_m()
    # System parameters
    params = SystemParams(
        R_sft=Rsft,
        R_icl=Ricl,
        p=p,
        q=0.2
    )
    num_p = 8
    diff_list = list(np.linspace(0.1, 20.0, 200))
    Ricl_list = list(np.linspace(1.0, 8.0, num_p))
    param1_list = diff_list
    param1_list = [float(param1) for param1 in param1_list]
    param2_list = Ricl_list
    param2_list = [float(param2) for param2 in param2_list]
    R_star_arrays = []
    count = 0
    for param2 in param2_list:
        print(f"Processing {count} of {len(param2_list)}")
        count += 1
        params.R_icl = param2
        R_star_list = []
        for param1 in param1_list:
            params.R_sft = param2 + param1
            type1.update_zeta()
            type2.update_zeta()
            calculator = GeneralMFECalculator(type1, type2, params, error_fn, congestion_fn)
            result = calculator.calculate_equilibrium(verbose=False)
            R_star_list.append(result['R_star'])
        R_star_arrays.append(R_star_list)

    plt.figure()
    for i in range(num_p):
        t = i / (num_p - 1) if num_p > 1 else 0
        color = (1 - t) * color_1 + t * color_2
        plt.plot(param1_list, R_star_arrays[i], label=f'$R_{{\mathrm{{ICL}}}}$={param2_list[i]:.0f}',linewidth=linewidth, color=color)
    plt.legend(fontsize=legend_font_size, frameon=True, fancybox=True, ncol=2)
    plt.xlabel(r'$R_{{\mathrm{{SFT}}}}-R_{{\mathrm{{ICL}}}}$', fontsize=font_size)
    plt.xticks([0, 4, 8, 12, 16, 20])
    plt.yticks([0, 4, 8, 12, 16, 20])
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.ylabel(r'$R^*$', fontsize=font_size)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/homo_general_linear_R_sft.pdf', dpi=300, bbox_inches='tight')
    

