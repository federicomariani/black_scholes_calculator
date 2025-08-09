import numpy as np
from scipy.stats import norm

def black_scholes(S, K, r, sigma, T, option_type="call", q = 0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        delta = np.exp(-q * T) * norm.cdf(d1)
        rho = np.exp(-q * T) * norm.cdf(d2)
        theta = (-S * np.exp(-q*T) * norm.pdf(d1) * sigma / (2*np.sqrt(T))
                 - r*K*np.exp(-r*T)*norm.cdf(d2)
                 + q*S*np.exp(-q*T)*norm.cdf(d1))    
    elif option_type.lower() == "put":
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        theta = (-S * np.exp(-q*T) * norm.pdf(d1) * sigma / (2*np.sqrt(T))
                 + r*K*np.exp(-r*T)*norm.cdf(-d2)
                 - q*S*np.exp(-q*T)*norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho
    }

from scipy.optimize import brentq 

def implied_volatility(market_price, S, K, r, T, option_type="call", q = 0.0, tol = 1e-6):
    def objective(sigma):
        return black_scholes(S, K, r, sigma, T, option_type, q)["price"] - market_price
    
    try:
        iv = brentq(objective, 1e-6, 5.0, xtol=tol)
    except ValueError:
        iv = np.nan  # If no solution is found, return NaN
    return iv