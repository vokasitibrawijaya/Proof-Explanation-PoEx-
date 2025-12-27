'use strict';

async function main() {
  const ccid = process.env.CHAINCODE_ID;
  const address = process.env.CHAINCODE_SERVER_ADDRESS || '0.0.0.0:9999';

  if (!ccid) {
    throw new Error('Missing CHAINCODE_ID env var (expected package ID)');
  }

  const Bootstrap = require('fabric-shim/lib/contract-spi/bootstrap');
  const modulePath = process.cwd();
  const { contracts, serializers, title, version } = Bootstrap.getInfoFromContract(modulePath);
  const fileMetadata = await Bootstrap.getMetadata(modulePath);

  await Bootstrap.register(
    contracts,
    serializers,
    fileMetadata,
    title,
    version,
    { ccid, address },
    true
  );
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
