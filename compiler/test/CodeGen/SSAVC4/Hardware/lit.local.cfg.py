# SSAVC4 hardware fixtures are verified by the M3 command/hardware wrappers,
# not by direct lit execution. Do not recurse into Hardware/Run as a lit suite.
config.excludes = ['Run']
