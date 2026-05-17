# SSAVC4 hardware fixtures are verified by M3 command/hardware wrappers,
# not direct lit execution. Do not recurse into Hardware/Run as a lit suite.
config.excludes = ['Run']
