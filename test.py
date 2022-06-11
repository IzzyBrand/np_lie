
"""test.py: test all functions in np_lie.py

Izzy Brand, 2022
"""

import numpy as np

from np_lie import *

###############################################################################
# SO3
###############################################################################

def SO3_hat_vee():
    w = np.random.randn(3)
    return np.allclose(w, SO3_vee(SO3_hat(w)))

def SO3_exp_log():
    w = np.random.randn(3)
    return np.allclose(w, SO3_log(SO3_exp(w)))

def SO3_Exp_Log():
    w = np.random.randn(3)
    S = SO3_hat(w)
    return np.allclose(S, SO3_Log(SO3_Exp(S)))

def SO3_left_jacobians_invert():
    w = np.random.randn(3)
    J_l = SO3_left_jacobian(w)
    J_l_inv = SO3_left_jacobian_inverse(w)
    return np.allclose(np.eye(3), J_l @ J_l_inv)

###############################################################################
# SE3
###############################################################################

def SE3_hat_vee():
    τ = np.random.randn(6)
    return np.allclose(τ, SE3_vee(SE3_hat(τ)))

def SE3_exp_log():
    τ = np.random.randn(6)
    return np.allclose(τ, SE3_log(SE3_exp(τ)))

def SE3_Exp_Log():
    τ = np.random.randn(6)
    τˆ = SE3_hat(τ)
    return np.allclose(τˆ, SE3_Log(SE3_Exp(τˆ)))

###############################################################################
# Main
###############################################################################

def main():
    tests = {
        "SO3 vee inverts hat": SO3_hat_vee,
        "SO3 log inverts exp": SO3_exp_log,
        "SO3 Log inverts Exp": SO3_Exp_Log,
        "SO3 left jacobians invert": SO3_left_jacobians_invert,
        "SE3 vee inverts hat": SE3_hat_vee,
        "SE3 log inverts exp": SE3_exp_log,
        "SE3 Log inverts Exp": SE3_Exp_Log,
    }

    for name, test_fn in tests.items():
        print("[PASSED]" if test_fn() else "[FAILED]", name)

if __name__ == "__main__":
    main()
