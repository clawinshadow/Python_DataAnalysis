import numpy as np
from numpy import poly1d

p = np.poly1d([2, -3, -5]) # construct the polynomial 2x**2 - 3x - 5
print(p)

print(p(1))  # Evaluate the polynomial at x = 1
print(p.r)  # Find the roots, means the roots of 2x**2 - 3x - 5 = 0
print(p(p.r))
print(p.c) # Show the coefficients
print(p.order) # Display the order (the leading zero-coefficients are removed)
print(p[0], p[1], p[2]) # Show the coefficient of the k-th power in the polynomial (which is equivalent to p.c[-(i+1)]):

print(p * p)
print(p ** 2)     # square of polynomial
print(np.square(p))  # square of individual coefficients

z = np.poly1d([1,2,3], variable='z') # variable controll the char to be printed
print(z)

# Construct a polynomial from its roots
p = np.poly1d([1, 2], True)
print(p)

print(p.deriv(m=1))  # 1st derivative 一阶导数
print(p.deriv(m=2))  # 2nd derivative 二阶导数
  
print(p.integ(m=1))  # indefinite integral  不定积分
