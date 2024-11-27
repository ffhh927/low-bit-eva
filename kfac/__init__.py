from kfac.eva import KFAC as EVA

from kfac.eva_8bit_cpu import KFAC as EVA8bit_CPU
from kfac.eva_8bit import KFAC as EVA8bit

from kfac.eva_fused import KFAC as EVA_Fused
from kfac.eva_8bit_fused import KFAC as EVA8bit_Fused


kfac_mappers = {
    'eva': EVA,
    'eva_fused': EVA_Fused,
    'eva_8bit': EVA8bit,
    'eva_8bit_cpu': EVA8bit_CPU,
    'eva_8bit_fused': EVA8bit_Fused
#    'kfac': KFAC,
#    'adasgd': ADASGD,
#    'adasgd2': ADASGD2,
#    'sam': SAM, 
    }

def get_kfac_module(kfac='eva'):
    return kfac_mappers[kfac]
