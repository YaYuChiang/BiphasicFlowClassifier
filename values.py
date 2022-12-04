import os
from lib.loglo import init_log

lo = init_log(__file__, save_log=True, level=os.environ.get("LOG_LEVEL", "0"))

gravity = 9.8
system_cross_sectional_area = 0.00000153938
characteristic_length = 0.0013

# flow rate
ul_per_min = 3000
m3_per_sec = ul_per_min * 1.6666666667E-11  # 0.000000050000000001
velocity = m3_per_sec / system_cross_sectional_area  # 0.03248061

feature_liquid = ['Viscosity (Pa·s)',
                  'Density (kg·m-3)',
                  'Polarity Index',
                  'to water (N·m-1)',
                  'to air (mN·m-1)']

class Liquid:
    def __init__(self, param) -> None:
        assert isinstance(param, list)
        self.viscosity = param[0]
        self.density = param[1]
        self.polarityIndex = param[2]
        self.toWater = param[3]
        self.toAir = param[4]

        self.arr = param.copy()

dict_liquid = {
    'toluene': Liquid([0.0006, 866, 2.4, 0.0000284, 26.96]),
    'heptane': Liquid([0.00042, 680, 0, 0.00002014, 33.35]),
    'butanol': Liquid([0.00295, 810, 4, 0.0000242, 1.84]),
    'ethyl acetate': Liquid([0.00045, 902, 4.4, 0.000024, 5.61]),
    'unknown': Liquid([1, 1, 1, 1, 1]),
}

class water:
    viscosity = 0.001
    density = 997
    polarity_index = 9

class CaWeMo:
    @staticmethod
    def get_liquid(key_liquid):
        if key_liquid not in dict_liquid.keys():
            lo.error(f"not a valid liquid name {key_liquid}")
            return
        return dict_liquid.get(key_liquid)

    @staticmethod
    def get_ca(liquid_name):
        liquid = CaWeMo.get_liquid(liquid_name)
        if liquid is None:
            return
        # =B$2*$D$14/B$5
        return liquid.viscosity * velocity / liquid.toWater
    
    @staticmethod
    def get_bo(liquid_name):  # aka 'we'
        liquid = CaWeMo.get_liquid(liquid_name)
        if liquid is None:
            return
        # =B3*($D$14^2)*$B$10/B5
        return liquid.density * pow(velocity, 2) * characteristic_length / liquid.toWater
        
    @staticmethod
    def get_mo(liquid_name):
        liquid = CaWeMo.get_liquid(liquid_name)
        if liquid is None:
            return
        # =$B$8*($F$2^4)/($F$3*B5^3)
        return gravity * pow(water.viscosity, 4) / (water.density * pow(liquid.toWater, 3))
    
    @staticmethod
    def get_polarity(liquid_name):
        liquid = CaWeMo.get_liquid(liquid_name)
        if liquid is None:
            return
        return liquid.polarityIndex

# experiment to method
dict_methods = {
    1: ['LR', 'SVC', 'XGB', 'DT', 'GB', 'RF', 'MLP'],
    2: ['LR', 'SVC', 'XGB', 'DT', 'GB', 'RF', 'ENS']
}

# print(list(dict_liquid.keys())[0])