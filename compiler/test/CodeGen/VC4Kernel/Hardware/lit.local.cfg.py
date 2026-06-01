# VC4Kernel hardware fixtures are verified by explicit candidate runners, not
# direct lit execution. Do not recurse into Hardware/Run as a lit suite.
config.excludes = ['Run']
