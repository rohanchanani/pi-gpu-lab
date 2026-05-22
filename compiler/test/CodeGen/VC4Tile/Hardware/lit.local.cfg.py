# M4 slice 03 installs the VC4Tile candidate hardware-run directory without
# making hardware fixtures active before an executable VC4Tile feature exists.
# Later vertical hardware slices may replace this local config when they add
# real Run/* fixtures and CPU-reference checks.
config.excludes = list(getattr(config, "excludes", [])) + ["Run"]
