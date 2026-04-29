
Candidate side disabled

Candidate/codegen execution is intentionally disabled for conv1d_3tap.

The reference side is the trusted hardware ground truth. After VC4 codegen can emit the qasm, launcher .c, and launcher .h bundle for this final-stage input, add a candidate runner that uses the same expected.json oracle.

Do not claim candidate coverage or update catalog.json for this test until the generated candidate side runs successfully on hardware.

