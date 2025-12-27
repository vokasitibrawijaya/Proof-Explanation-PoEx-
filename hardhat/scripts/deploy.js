const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("Deploying FedXChain contract...");

  const FedXChain = await hre.ethers.getContractFactory("FedXChain");
  const fedxchain = await FedXChain.deploy();

  await fedxchain.waitForDeployment();
  
  const contractAddress = await fedxchain.getAddress();
  console.log("FedXChain deployed to:", contractAddress);

  // Get ABI for Python clients
  const artifact = await hre.artifacts.readArtifact("FedXChain");
  
  // Save deployment info with ABI
  const deployment = {
    contractAddress: contractAddress,
    abi: artifact.abi,
    network: hre.network.name,
    deployedAt: new Date().toISOString()
  };

  // Save to multiple locations
  const deploymentsDir = path.join(__dirname, "../deployments");
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir, { recursive: true });
  }

  fs.writeFileSync(
    path.join(__dirname, "../deployment.json"),
    JSON.stringify(deployment, null, 2)
  );
  
  fs.writeFileSync(
    path.join(deploymentsDir, "contract_address.json"),
    JSON.stringify(deployment, null, 2)
  );

  console.log("Deployment info saved to deployment.json and deployments/contract_address.json");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
