# Harrison Hidalgo
# MAE 6760 - Final Project
# This file holds our physical variables.

import math

## Ball Radius
r = 0.07 
## Ball Area
ball_area = math.pi*r**2
## Air Density
rho = 1.229
## Gravity
g = -9.81
## Air Friction Coefficient
coefficient = 0.5 # ~Re of 10,000
c = 0.5 * rho * ball_area * coefficient
## Ball Mass
m = 1.0
