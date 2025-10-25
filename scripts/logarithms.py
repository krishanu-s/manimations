"""Depict the following ideas:

Logarithms. Why are they useful?

Geometric series
- (1-r) + r(1-r) + r^2(1-r) + ... = 1
- Deduce that 1 + r + r^2 + ... = 1 / (1-r)
- Partial version is 1 + r + r^2 + ... + r^{n-1} = (1-r^n)/(1-r)

Quadrature of a parabola y = x^2
- Goal is to calculate the area from x=0 to x=1. Call this A.
- Quadrature under a curve scales under horizontal and vertical stretches.
  - See this with a linear function: area of a triangle.
- Therefore, the area from x=0 to x=2 is 8A.
- This area can also be divided up into four parts of areas A, 1, 1, A. Thus,
  8A = 2A + 2, so A = 1/3.

Quadrature of y = x^3
- List the three we have so far on the right.
- Do a similar calculation as for y = x^2.
- Suggest the formula for y = x^n. Use the same argument to get
  A_n + sum_{k=0}^{n}A_k * (nCk) = 2^{n+1}A_n
  Then use induction and the fact that (nCk)/(k+1) = ((n+1)C(k+1))/(n+1)
  to obtain A_n = 1/(n+1).

Quadrature of a hyperbola y = 1/x
- The natural logarithm ln(a) as the area under the curve y = 1/x from x=1 to x=a
- ln(a) also equals the area from x=b to x=ab, for any b.
- This implies ln(a) + ln(b) = ln(ab)
- This also implies ln(a^n) = n*ln(a)

How would you calculate the logarithm?
- ln(1+a) is the area under 1 / (1 + x) from 0 to a.
- 1 / (1+x) = 1 - x + x^2 - x^3 + ... ()
- And we already calculated the area under each of these.
- So a - a^2/2 + a^3/3 - ..., so long as a <= 1."""