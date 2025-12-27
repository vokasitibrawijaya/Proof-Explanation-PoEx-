'use strict';

const { Contract } = require('fabric-contract-api');

function _toFloatArray(arr) {
  if (!Array.isArray(arr)) {
    throw new Error('expected array');
  }
  return arr.map((x) => {
    const v = Number(x);
    if (!Number.isFinite(v)) throw new Error('non-finite value in vector');
    return v;
  });
}

function _normalizeToProb(vec, epsilon = 1e-10) {
  const absVec = vec.map((v) => Math.abs(v));
  const smoothed = absVec.map((v) => v + epsilon);
  const sum = smoothed.reduce((a, b) => a + b, 0);
  if (!(sum > 0)) throw new Error('invalid normalization sum');
  return smoothed.map((v) => v / sum);
}

function _kl(p, q) {
  if (p.length !== q.length) throw new Error('vector length mismatch');
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    s += p[i] * Math.log(p[i] / q[i]);
  }
  return s;
}

class PoExContract extends Contract {
  async InitLedger(ctx) {
    const threshold = 0.5;
    const state = {
      threshold,
      globalExplanation: null,
      trust: {
        '0': 1.0,
        '1': 1.0,
        '2': 1.0
      }
    };
    await ctx.stub.putState('poex:state', Buffer.from(JSON.stringify(state)));
    return JSON.stringify({ ok: true, threshold });
  }

  async SetThreshold(ctx, thresholdStr) {
    const threshold = Number(thresholdStr);
    if (!Number.isFinite(threshold) || threshold < 0) {
      throw new Error('invalid threshold');
    }
    const state = await this._getState(ctx);
    state.threshold = threshold;
    await ctx.stub.putState('poex:state', Buffer.from(JSON.stringify(state)));
    return JSON.stringify({ ok: true, threshold });
  }

  async UpdateGlobalExplanation(ctx, explanationJson) {
    const vec = _toFloatArray(JSON.parse(explanationJson));
    const state = await this._getState(ctx);
    state.globalExplanation = vec;
    await ctx.stub.putState('poex:state', Buffer.from(JSON.stringify(state)));
    return JSON.stringify({ ok: true, len: vec.length });
  }

  async SubmitUpdate(ctx, clientId, roundStr, modelHash, explanationJson) {
    const round = parseInt(roundStr, 10);
    if (!Number.isFinite(round) || round < 0) throw new Error('invalid round');

    const explanation = _toFloatArray(JSON.parse(explanationJson));
    const state = await this._getState(ctx);

    const threshold = Number(state.threshold ?? 0.5);
    const trust = Number((state.trust && state.trust[clientId]) ?? 1.0);

    let accepted = true;
    let nsds = 0.0;

    if (state.globalExplanation && Array.isArray(state.globalExplanation)) {
      const p = _normalizeToProb(explanation);
      const q = _normalizeToProb(_toFloatArray(state.globalExplanation));
      nsds = _kl(p, q);
      accepted = nsds <= threshold;
    }

    const newTrust = accepted ? Math.min(1.0, trust + 0.01) : Math.max(0.0, trust - 0.1);
    state.trust = state.trust || {};
    state.trust[clientId] = newTrust;

    const updateKey = `poex:update:${round}:${clientId}`;
    const update = {
      clientId,
      round,
      modelHash,
      nsds,
      accepted,
      trustBefore: trust,
      trustAfter: newTrust,
      ts: new Date().toISOString()
    };

    await ctx.stub.putState(updateKey, Buffer.from(JSON.stringify(update)));
    await ctx.stub.putState('poex:state', Buffer.from(JSON.stringify(state)));

    ctx.stub.setEvent('PoExUpdate', Buffer.from(JSON.stringify(update)));
    return JSON.stringify(update);
  }

  async GetState(ctx) {
    const state = await this._getState(ctx);
    return JSON.stringify(state);
  }

  async GetUpdate(ctx, roundStr, clientId) {
    const round = parseInt(roundStr, 10);
    const key = `poex:update:${round}:${clientId}`;
    const b = await ctx.stub.getState(key);
    if (!b || b.length === 0) throw new Error('not found');
    return b.toString('utf8');
  }

  async _getState(ctx) {
    const b = await ctx.stub.getState('poex:state');
    if (!b || b.length === 0) {
      return { threshold: 0.5, globalExplanation: null, trust: {} };
    }
    return JSON.parse(b.toString('utf8'));
  }
}

module.exports = PoExContract;
