LAYERS_NAME = ["sequence", "ssqa", "transitions"]
PROFILE_HEADER = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                  'M->M', 'M->I', 'M->D', 'I->M', 'I->I',
                  'D->M', 'D->D', 'Neff', 'Neff_I', 'Neff_D')  # yapf: disable
AMINO_ACIDS = AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = AA_IDS = {k: i for i, k in enumerate(AA)}
AA_MAT = None
device = "cpu"
DATA = "/home/malbranke/data/"

# Dictionary to convert 'secStructList' codes to DSSP values
# https://github.com/rcsb/mmtf/blob/master/spec.md#secstructlist
sec_struct_codes = "GHIBESTC"

abc_codes = {"a": 0, "b": 1, "c": 2}
dssp_codes = {"G": 0,
               "H": 1,
               "I": 2,
               "B": 3,
               "E": 4,
               "S": 5,
               "T": 6,
               "C": 7}

# Converter for the DSSP secondary pattern elements
# to the classical ones
dssp_to_abc = {"G": "a",
               "H": "a",
               "I": "a",
               "B": "b",
               "E": "b",
               "S": "c",
               "T": "c",
               "C": "c"}

pdb_codes = {0: "I",
                    1: "S",
                    2: "H",
                    3: "E",
                    4: "G",
                    5: "B",
                    6: "T",
                    7: "C"}

I = (lambda x: x)

def ss8_to_ss3(x):
    if x <= 2:
        return 0
    if x >= 5:
        return 2
    return 1


# Converter for the DSSP secondary pattern elements
# to the classical ones
