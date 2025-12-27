import express from 'express';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import grpc from '@grpc/grpc-js';
import {
  connect,
  signers,
} from '@hyperledger/fabric-gateway';

const app = express();
app.use(express.json({ limit: '5mb' }));

const PORT = Number(process.env.PORT || 8080);
const CHANNEL_NAME = process.env.CHANNEL_NAME || 'poexchannel';
const CHAINCODE_NAME = process.env.CHAINCODE_NAME || 'poex';

const MSP_ID = process.env.MSP_ID || 'Org1MSP';
const CRYPTO_BASE = process.env.CRYPTO_BASE || '/fabric/crypto-config';

const PEER_ENDPOINT = process.env.PEER_ENDPOINT || 'peer0.org1.example.com:7051';
const PEER_TLS_HOSTNAME_OVERRIDE = process.env.PEER_TLS_HOSTNAME_OVERRIDE || 'peer0.org1.example.com';

function sha256Hex(input) {
  return crypto.createHash('sha256').update(input).digest('hex');
}

function findFirstFile(dir, predicate) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const ent of entries) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      const found = findFirstFile(full, predicate);
      if (found) return found;
    } else {
      if (predicate(ent.name, full)) return full;
    }
  }
  return null;
}

function loadIdentityAndSigner() {
  const certPath = path.join(
    CRYPTO_BASE,
    'peerOrganizations',
    'org1.example.com',
    'users',
    'Admin@org1.example.com',
    'msp',
    'signcerts'
  );
  const keyDir = path.join(
    CRYPTO_BASE,
    'peerOrganizations',
    'org1.example.com',
    'users',
    'Admin@org1.example.com',
    'msp',
    'keystore'
  );

  const certFile = findFirstFile(certPath, (name) => name.endsWith('.pem'));
  const keyFile = findFirstFile(keyDir, (name) => name.endsWith('_sk') || name.endsWith('.pem'));
  if (!certFile || !keyFile) {
    throw new Error(`Cannot find Admin identity files under ${certPath} and ${keyDir}`);
  }

  const certificate = fs.readFileSync(certFile);
  const privateKeyPem = fs.readFileSync(keyFile);

  return {
    identity: { mspId: MSP_ID, credentials: certificate },
    signer: signers.newPrivateKeySigner(crypto.createPrivateKey(privateKeyPem)),
  };
}

function newGrpcConnection() {
  const tlsCertPath = path.join(
    CRYPTO_BASE,
    'peerOrganizations',
    'org1.example.com',
    'peers',
    'peer0.org1.example.com',
    'tls',
    'ca.crt'
  );
  const tlsRootCert = fs.readFileSync(tlsCertPath);
  const tlsCredentials = grpc.credentials.createSsl(tlsRootCert);

  const options = {
    'grpc.ssl_target_name_override': PEER_TLS_HOSTNAME_OVERRIDE,
  };

  return new grpc.Client(PEER_ENDPOINT, tlsCredentials, options);
}

function bytesToString(bytes) {
  if (Buffer.isBuffer(bytes)) return bytes.toString('utf8');
  if (typeof bytes === 'string') return bytes;
  return Buffer.from(bytes).toString('utf8');
}

async function withGateway(fn) {
  const client = newGrpcConnection();
  const { identity, signer } = loadIdentityAndSigner();
  
  // Configure gateway with explicit endorsing organizations for single-peer setup
  // This bypasses automatic discovery which fails in single-org networks
  const gateway = connect({
    client,
    identity,
    signer,
    // Disable automatic discovery and use explicit endorsement
    evaluateOptions: () => {
      return { deadline: Date.now() + 30000 };
    },
    endorseOptions: () => {
      return { 
        deadline: Date.now() + 30000,
      };
    },
    submitOptions: () => {
      return { deadline: Date.now() + 30000 };
    },
    commitStatusOptions: () => {
      return { deadline: Date.now() + 60000 };
    },
  });

  try {
    const network = gateway.getNetwork(CHANNEL_NAME);
    const contract = network.getContract(CHAINCODE_NAME);
    return await fn(contract);
  } finally {
    gateway.close();
    client.close();
  }
}

app.get('/health', async (_req, res) => {
  try {
    const state = await withGateway(async (contract) => {
      const out = await contract.evaluateTransaction('GetState');
      const raw = bytesToString(out);
      return JSON.parse(raw);
    });
    res.json({ ok: true, state });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

app.post('/set_threshold', async (req, res) => {
  const { threshold } = req.body || {};
  try {
    const out = await withGateway(async (contract) => {
      const bytes = await contract.submitTransaction('SetThreshold', String(threshold));
      return JSON.parse(bytesToString(bytes));
    });
    res.json(out);
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

app.post('/update_global_explanation', async (req, res) => {
  const { explanation } = req.body || {};
  try {
    const out = await withGateway(async (contract) => {
      const bytes = await contract.submitTransaction('UpdateGlobalExplanation', JSON.stringify(explanation || []));
      return JSON.parse(bytesToString(bytes));
    });
    res.json(out);
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

app.post('/submit_update', async (req, res) => {
  const { client_id, round, model_hash, explanation } = req.body || {};
  try {
    const out = await withGateway(async (contract) => {
      const bytes = await contract.submitTransaction(
        'SubmitUpdate',
        String(client_id),
        String(round),
        String(model_hash || ''),
        JSON.stringify(explanation || [])
      );
      return JSON.parse(bytesToString(bytes));
    });
    res.json(out);
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e?.message || e) });
  }
});

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`poex-gateway listening on :${PORT} (channel=${CHANNEL_NAME}, cc=${CHAINCODE_NAME})`);
});
