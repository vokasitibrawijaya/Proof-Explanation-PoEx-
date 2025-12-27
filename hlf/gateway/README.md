# PoEx Gateway (HLF)

This folder will host a small gateway service that can submit Fabric transactions (PoEx chaincode) and return the chaincode response to the Python aggregator.

Planned approach:
- Use Fabric Gateway SDK (Node.js) to submit `SubmitUpdate` and `UpdateGlobalExplanation`.
- Mount Org1 Admin identity from `hlf/crypto-config`.

(Scaffold only; implementation is added in next step.)
