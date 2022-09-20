from gekko import GEKKO
import numpy as np


# Create Gekko model
# Steel rod temperature profile
# Diameter = 3 cm
# Length = 10 cm


class HRModel(GEKKO):
    def __init__(self):
        super().__init__(remote=False)
        seg = 100  # number of segments
        T_melt = 1426  # melting temperature of H13 steel
        pi = 3.14159  # pi
        d = 3 / 100  # rod diameter (m)
        L = 10 / 100  # rod length (m)
        L_seg = L / seg  # length of a segment (m)
        A = pi * d ** 2 / 4  # rod cross-sectional area (m)
        As = pi * d * L_seg  # surface heat transfer area (m^2)
        heff = 5.8  # heat transfer coeff (W/(m^2*K))
        keff = 28.6  # thermal conductivity in H13 steel (W/m-K)
        rho = 7760  # density of H13 rod steel (kg/m^3)
        cp = 460  # heat capacity of H13 steel (J/kg-K)
        Ts = 23  # temperature of the surroundings (°C)
        c2k = 273.15  # Celcius to Kelvin

        t_final = 600000
        nt = 200
        tsim = np.linspace(0, t_final, nt)
        self.dt = tsim[1] - tsim[0]
        self.time = [0, self.dt]
        self.Qh = self.MV(ub=T_melt)  # heater temperature (°C)

        self.T = [self.Var(23) for i in range(seg)]  # temperature of the segments (°C)

        # Energy balance for the rod (segments)
        # accumulation =
        #    (heat gained from upper segment)
        #  - (heat lost to lower segment)
        #  - (heat lost to surroundings)
        # Units check
        # kg/m^3 * m^2 * m * J/kg-K * K/sec =
        #     W/m-K   * m^2 *  K / m
        #  -  W/m-K   * m^2 *  K / m
        #  -  W/m^2-K * m^2 *  K

        # first segment
        self.Equation(rho * A * L_seg * cp * self.T[0].dt() == self.Qh \
                      - keff * A * (self.T[0] - self.T[1]) / L_seg \
                      - heff * As * (self.T[0] - Ts))
        # middle segments
        self.Equations([rho * A * L_seg * cp * self.T[i].dt() == \
                        keff * A * (self.T[i - 1] - self.T[i]) / L_seg \
                        - keff * A * (self.T[i] - self.T[i + 1]) / L_seg \
                        - heff * As * (self.T[i] - Ts) for i in range(1, seg - 1)])
        # last segment
        self.Equation(rho * A * L_seg * cp * self.T[seg - 1].dt() == \
                      keff * A * (self.T[seg - 2] - self.T[seg - 1]) / L_seg \
                      - heff * (As + A) * (self.T[seg - 1] - Ts))

        # simulation
        self.options.IMODE = 4

    def run(self, Q, disp=False):
        self.Qh.VALUE = Q
        self.solve(disp=disp)
        return self.T.copy()[-1][-1]
