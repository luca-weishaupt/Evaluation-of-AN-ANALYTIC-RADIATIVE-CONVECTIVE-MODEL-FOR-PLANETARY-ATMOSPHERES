import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
from scipy.optimize import fsolve
from matplotlib.patches import Rectangle
import os
np.seterr(divide='ignore', invalid='ignore')

def inc_upper_gamma(a,x):
    """
    upper incomplete gamma function (Appendix: A9)
    :param a: float
    :param x: float
    :return: float
    """
    # Using a conditional because sc.gammaincc cannot handle a = 0
    return sc.exp1(x) if a == 0 else sc.gammaincc(a, x) * sc.gamma(a)


def eq18(pp, kD, tau_nought=2.0, n=2.0, gamma=1.4, D=2.0):
    """
    equation 18 solved for the normalized temperature in terms of k/D and p/p_0 for fig3
    :param pp: float
    normalized pressure p / p_0
    :param kD: [float]
    k/D
    :param tau_nought: [float]
    tau_nought
    :param n: [float]
    scaling factor
    :param D: [float]
    diffusivity factor. Often taken as 1.66, 3/2, or 2
    :param gamma: [float]
    ratio of specific heats
    :return: [float]
    normalized temperature sigma T^4 / F
    """
    # form eq 6 we have tau = tau_0 (p/p_0)^n
    tau = tau_nought*np.power(pp,n)
    # Using eq32 and 33 to check for instability
    LHS = (n*D*kD*tau/4.0)*((D**2-(D*kD)**2)*np.exp(-D*kD*tau))/(D*kD*D+D**2+((D*kD)**2-D**2)*np.exp(-D*kD*tau))
    instability = LHS>(gamma-1)/gamma

    return 0.5*(1+1/kD+(kD-1/kD)*np.exp(-D*kD*tau)), instability

def eq24(pp, tau_nought=2.0, n=2, D=2.0, gamma=1.4):

    # form eq 6 we have tau = tau_0 (p/p_0)^n

    tau = tau_nought * np.power(pp, n)

    # Using eq32 to check for instability
    instability = (D * n * tau) / (4 * (1 + D * tau)) > (gamma - 1) / gamma

    return (1+D*tau)/2, instability

def eq30(four_beta_n, tau_rc, i_tau_nought, D=2.0):
    """
    Equation 30, describing the radiative-convective boundary in "the Simplest Radiative–Convective Model"
    :param four_beta_n: float
    The value of 4 \beta / n
    :param tau_rc: float
    The value of \tau_rc
    :param i_tau_nought: float
    The initial guess for \tau_0
    :param D: float
     diffusivity factor. Often taken as 1.66, 3/2, or 2
    :return: float
    The value of tau_0
    """
    equ30 = lambda tau_nought : np.power(tau_nought / tau_rc, four_beta_n) * \
                               np.exp(-D * (tau_nought - tau_rc)) * \
                               (1 + np.exp(D * tau_nought) / np.power(D * tau_nought, four_beta_n) *
                                (inc_upper_gamma(1 + four_beta_n, D * tau_rc) -
                                 inc_upper_gamma(1 + four_beta_n, D * tau_nought))) \
                               - (2 + D * tau_rc) / (1 + D * tau_rc)
    return fsolve(equ30, i_tau_nought)[0]


def eq31(four_beta_n, i_tau_rc, D=2.0):
    equ31 = lambda tau_rc: inc_upper_gamma(1 + four_beta_n, D * tau_rc) / (np.power((D * tau_rc), four_beta_n) * np.exp(-D * tau_rc)) - (2 + D * tau_rc) / (1 + D * tau_rc)
    return fsolve(equ31, i_tau_rc)[0]

def eq32(four_beta_n, D=2.0):
    return four_beta_n/((1-four_beta_n)*D)

def venusrad(p, n, tau_nought, F=160.0, D=2.0, p_nought=92):

    return np.power((F/(2*5.67*10**(-8)))*(1+D*tau_nought*np.power(p/p_nought,n)),1/4)

def venusconv(p, n, tau_rc, tau_nought, beta, F=160.0, D=2.0, p_nought=92):

    return np.power(F/(2*5.67*10**(-8))*(1+D*tau_rc),1/4)*np.power(tau_nought/ tau_rc,beta/n)*np.power(p/p_nought,beta)

def earthrad(p, n, tau_nought, F=400, D=2.0, p_nought=1):

    return np.power((F/(2*5.67*10**(-8)))*(1+D*tau_nought*np.power(p/p_nought,n)),1/4)

def earthconv(p, n, tau_rc, tau_nought, beta, F=400, D=2.0, p_nought=1):

    return np.power(F/(2*5.67*10**(-8))*(1+D*tau_rc),1/4)*np.power(tau_nought/ tau_rc,beta/n)*np.power(p/p_nought,beta)


def fig1(tau_rc=None, lower_limits=None, original=True, show=True, dest=None,  D=2.0):
    """
    Plot Figure 1
    :param tau_rc: iterable
    List of values of tau_rc to be plotted
    :param lower_limits: iterable
    List of values of 4 \beta / n when D \tau_0 goes to infinity (diverges)
    :param original: boolean
    Will plot original plot from the paper if True, used tau_rc and lower_limits if False
    :param D: float
    Often taken as 1.66, 3/2, or 2
    :param show: boolean
    Show the plot if True.
    :param dest: String
    File path to destination folder and filename
    :return: None
    Plots
    """
    if original:
        tau_rcs = [0.005, 0.05, 0.25, 0.5, 1.0]
        lower_lims = [0.2, 0.302, 0.506, 0.6189, 0.731]
    else:
        tau_rcs = tau_rc
        lower_lims = lower_limits

    fig, ax = plt.subplots()

    # Show the region that is typically found in the solar system
    rect = Rectangle((0.3, 0), 0.2, 30, alpha=0.2, label="typical values in the solar system")
    ax.add_patch(rect)

    # Plot D tau_0 against 4 \beta / n for various values of D \tau_rc
    for i in range(len(tau_rcs)):
        # values for 4 \beta / n
        four_beta_ns = np.linspace(lower_lims[i], 1, 1000)
        sols = []
        for four_beta_n in four_beta_ns:
            sols.append(eq30(four_beta_n, tau_rcs[i], 0.1 / D, D))
        ax.plot(four_beta_ns, sols, label="$D τ_{rc}=$" + "{}".format(tau_rcs[i] * D))


    plt.legend()
    plt.yscale('log')
    plt.xlim([0.2, 1.0])
    plt.ylim([0.005, 20])
    plt.xlabel("4 β/n")
    plt.ylabel("reference optical depth,\n D $τ_0$")
    plt.gca().invert_yaxis()

    if not dest == None:
        plt.savefig(dest)

    if show:
        plt.show()

def fig2(original=True, show=True, dest=None, D=2):

    fig, ax = plt.subplots()

    # Show the region that is typically found in the solar system
    rect = Rectangle((0.3, 0), 0.2, 10, alpha=0.2, label="typical values in the solar system")
    ax.add_patch(rect)

    four_beta_ns = np.linspace(0.2, 1,100)
    sols = []
    for four_beta_n in four_beta_ns:
        sols.append(eq31(four_beta_n, 0.001 / D, D))

    ax.plot(four_beta_ns, sols, label="This Work")
    ax.plot(four_beta_ns, eq32(four_beta_ns, D), label="Sagan (1969)")

    plt.legend()
    plt.yscale('log')
    plt.xlim([0.2, 1.0])
    plt.ylim([0.01, 10])
    plt.xlabel("4 β/n")
    plt.ylabel("rad-conv boundary optical depth,\n D $τ_0$")
    plt.gca().invert_yaxis()

    if not dest == None:
        plt.savefig(dest)

    if show:
        plt.show()

def fig3(original=True, show = True, dest=None):

    # colors
    cs = iter(('r', 'g', 'b', 'm'))
    fig, ax = plt.subplots()

    # Normalized pressures
    pps = np.logspace(np.log(0.1), np.log(2), 1000)

    # In the case with k/D = 0 we use eq24
    unstable_pps = []
    for pp in pps:
        if eq24(pp)[1]: unstable_pps.append(pp)
    c = next(cs)
    ax.plot(eq24(pps)[0], pps, c=c, label="k=0 (no attenuation)")
    ax.plot(eq24(unstable_pps)[0], unstable_pps, c=c, lw=5, label="k=0 instability")

    # The other cases
    kDs = [0.1,0.5,10]
    for kD in kDs:
        unstable_pps = []
        for pp in pps:
            if eq18(pp, kD)[1]: unstable_pps.append(pp)
        c = next(cs)
        ax.plot(eq18(pps,kD)[0], pps, c=c, label="k/D={}".format(kD))
        if len(unstable_pps)>0:
            ax.plot(eq18(unstable_pps,kD)[0], unstable_pps, c=c, lw=5, label="k/D={}".format(kD))

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([0.4, 10])
    plt.ylim([0.1, 2])
    plt.xlabel(r"$\sigma T(p)^4 / F$"+"\n normalized temperature")
    plt.ylabel("normalized pressure,\n p/$p_0$")
    plt.gca().invert_yaxis()

    if not dest == None:
        plt.savefig(dest)

    if show:
        plt.show()

def fig8(original=True, show = True, dest=None):

    fig, ax = plt.subplots()

    rad_pressures = [np.linspace(0.01,0.2,1000),np.linspace(0.01,0.07,1000)]
    conv_pressures = [np.linspace(0.2,100,1000),np.linspace(0.07,100, 1000)]
    ns = [1,2]
    tau_noughts = [400, 2*10**5]
    tau_rcs = [1,0.1]
    beta = 0.8*(1.3-1)/1.3
    linestyles = [':',"--"]
    cs = ['r', 'b']

    for i in range(2):
        ax.plot(venusrad(rad_pressures[i], ns[i], tau_noughts[i]), rad_pressures[i], label="Model: n = {}".format(ns[i]), linestyle = linestyles[i], c=cs[i],lw=2)
        ax.plot(venusconv(conv_pressures[i],ns[i],tau_rcs[i],tau_noughts[i],beta),conv_pressures[i], linestyle = linestyles[i], c=cs[i], lw=2)

    plt.legend()
    plt.yscale('log')
    plt.xlabel("Temperature [K]")
    plt.ylabel("Pressure [bar]")
    plt.gca().invert_yaxis()

    if not dest == None:
        plt.savefig(dest)

    if show:
        plt.show()

def figEarth(original=True, show = True, dest=None):

    fig, ax = plt.subplots()

    p_c = 0.2

    rad_pressure = np.linspace(0.01,p_c,1000)
    conv_pressure = np.linspace(p_c,1,1000)
    n = 1
    tau_nought = 3
    tau_rc = 2/3
    beta = 0.8*(7/5-1)/(7/5)

    ax.plot(earthrad(rad_pressure, n, tau_nought), rad_pressure, c='r')
    ax.plot(earthconv(conv_pressure,n,tau_rc,tau_nought,beta),conv_pressure,  c='b')

    # plt.legend()
    plt.yscale('log')
    plt.xlabel("Temperature [K]")
    plt.ylabel("Pressure [bar]")
    plt.gca().invert_yaxis()

    if not dest == None:
        plt.savefig(dest)

    if show:
        plt.show()

if __name__ == "__main__":
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    fig1(dest="./figures/fig1.png")
    fig2(dest="./figures/fig2.png")
    fig3(dest="./figures/fig3.png")
    fig8(dest="./figures/fig8.png")
    figEarth(dest="./figures/earth.png")