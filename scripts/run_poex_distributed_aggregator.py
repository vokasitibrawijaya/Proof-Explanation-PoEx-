#!/usr/bin/env python3
"""PoEx Distributed FL Aggregator (HLF-backed)

This variant follows eksperimen_pox.md:
- Hyperledger Fabric v2.x network with PoEx chaincode
- Clients submit local model + SHAP explanation
- Aggregator asks PoEx chaincode to accept/reject updates before aggregation

Baseline mode: POEX_ENABLED=0 (accept all, FedAvg only).
"""

import base64
import csv
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import requests
import yaml
from flask import Flask, jsonify, request
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


class PoExGateway:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def set_threshold(self, threshold: float) -> dict:
        r = requests.post(f"{self.base_url}/set_threshold", json={"threshold": float(threshold)}, timeout=15)
        r.raise_for_status()
        return r.json()

    def update_global_explanation(self, explanation: np.ndarray) -> dict:
        r = requests.post(
            f"{self.base_url}/update_global_explanation",
            json={"explanation": [float(x) for x in explanation.tolist()]},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def submit_update(self, client_id: int, round_num: int, model_hash: str, explanation: np.ndarray, max_retries: int = 3) -> dict:
        payload = {
            "client_id": int(client_id),
            "round": int(round_num),
            "model_hash": str(model_hash),
            "explanation": [float(x) for x in explanation.tolist()],
        }
        import json
        
        # Retry logic for MVCC conflicts
        for attempt in range(max_retries):
            try:
                print(f"[GATEWAY-REQ] Sending to {self.base_url}/submit_update: {json.dumps(payload)[:200]}...")
                r = requests.post(
                    f"{self.base_url}/submit_update",
                    json=payload,
                    timeout=30,
                )
                print(f"[GATEWAY-RESP] Status={r.status_code}, Body={r.text[:500]}...")
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError as e:
                # Check for MVCC conflict
                if "MVCC_READ_CONFLICT" in str(e.response.text if hasattr(e, 'response') else ''):
                    if attempt < max_retries - 1:
                        backoff = (attempt + 1) * 2  # 2s, 4s, 6s
                        print(f"[RETRY] MVCC conflict for client {client_id}, retrying in {backoff}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(backoff)
                        continue
                raise  # Re-raise if not MVCC or last attempt


class PoExDistributedAggregator:
    def __init__(self):
        self.config = self._load_config()

        self.n_clients = int(os.getenv("N_CLIENTS", self.config.get("n_clients", 3)))
        self.max_rounds = int(os.getenv("MAX_ROUNDS", self.config.get("rounds", 10)))
        self.agg_method = os.getenv("AGG_METHOD", self.config.get("agg_method", "fedavg")).strip().lower()

        self.poex_enabled = _env_bool("POEX_ENABLED", default=True)
        self.poex_threshold = float(os.getenv("POEX_THRESHOLD", self.config.get("poex_threshold", 0.5)))
        self.poex_gateway_url = os.getenv("POEX_GATEWAY_URL", "http://poex-gateway:8080")
        self.gateway = PoExGateway(self.poex_gateway_url)

        self.run_id = os.getenv("RUN_ID", str(self.config.get("run_id", "1")))
        self.attack_type = os.getenv("ATTACK_TYPE", self.config.get("attack_type", "none")).strip().lower()
        self.malicious_ratio = float(os.getenv("MALICIOUS_RATIO", self.config.get("malicious_ratio", 0.0)))
        self.malicious_clients = os.getenv("MALICIOUS_CLIENTS", self.config.get("malicious_clients", "")).strip()

        self.output_dir = Path(os.getenv("OUTPUT_DIR", "/app/results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.output_dir / "poex_results.csv"
        self.clear_results = _env_bool("CLEAR_RESULTS", default=False)
        if self.clear_results and self.results_file.exists():
            self.results_file.unlink(missing_ok=True)

        self.current_round = 0
        self.clients_ready: dict[int, bool] = {}
        self.client_updates: dict[int, dict] = {}

        self.global_model = None
        self.X_test = None
        self.y_test = None
        self.client_data = None

        self._init_dataset()

        # Ensure gateway reachable if enabled
        if self.poex_enabled:
            for _ in range(30):
                try:
                    self.gateway.health()
                    break
                except Exception:
                    time.sleep(2)
            else:
                raise RuntimeError("PoEx gateway not reachable")

            # Wait for orderer Raft leader election (critical for transaction processing)
            print("⏳ Waiting 15s for orderer Raft leader election...")
            time.sleep(15)
            print("✓ Orderer should be ready")

            # Configure chaincode threshold
            try:
                self.gateway.set_threshold(self.poex_threshold)
            except Exception as e:
                print(f"! Failed to set PoEx threshold (continuing): {e}")

        print("✓ PoEx Aggregator initialized")
        print(f"  - Clients: {self.n_clients} | Rounds: {self.max_rounds}")
        print(f"  - Method: {self.agg_method} | PoEx enabled: {self.poex_enabled} | Threshold: {self.poex_threshold}")
        print(f"  - Gateway: {self.poex_gateway_url}")
        print(f"  - Run ID: {self.run_id} | Attack: {self.attack_type} | Malicious ratio: {self.malicious_ratio} | Malicious clients: {self.malicious_clients or '-'}")

    def _load_config(self) -> dict:
        config_path = os.getenv("CONFIG_PATH", "/app/configs/experiment_config.yaml")
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"! Failed to load config {config_path}: {e}")
        return {}

    def _init_dataset(self):
        seed = int(os.getenv("SEED", self.config.get("seed", 42)))
        n_samples = int(os.getenv("N_SAMPLES", self.config.get("n_samples", 500)))
        n_features = int(os.getenv("N_FEATURES", self.config.get("n_features", 20)))
        test_size = float(os.getenv("TEST_SIZE", self.config.get("test_size", 0.2)))
        n_informative = int(os.getenv("N_INFORMATIVE", self.config.get("n_informative", max(2, int(n_features * 0.7)))))
        n_redundant = int(os.getenv("N_REDUNDANT", self.config.get("n_redundant", max(0, int(n_features * 0.2)))))

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=2,
            n_informative=n_informative,
            n_redundant=n_redundant,
            random_state=seed,
        )
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        self.X_test = X_test
        self.y_test = y_test

        self.client_data = self._split_data_non_iid(X_train, y_train, seed=seed)

        self.global_model = SGDClassifier(loss="log_loss", max_iter=100, random_state=seed)
        self.global_model.fit(X_train[:100], y_train[:100])

    def _split_data_non_iid(self, X, y, seed: int = 42):
        rng = np.random.default_rng(seed)
        K = len(np.unique(y))
        N = len(y)
        alpha = float(os.getenv("DIRICHLET_ALPHA", self.config.get("dirichlet_alpha", 0.5)))
        min_required = int(os.getenv("MIN_SAMPLES_PER_CLIENT", self.config.get("min_samples_per_client", 10)))

        min_size = 0
        while min_size < min_required:
            idx_batch = [[] for _ in range(self.n_clients)]
            for k in range(K):
                idx_k = np.where(y == k)[0]
                rng.shuffle(idx_k)
                proportions = rng.dirichlet(np.repeat(alpha, self.n_clients))
                proportions = np.array([p * (len(idx_j) < N / self.n_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, cuts))]
            min_size = min(len(idx_j) for idx_j in idx_batch)

        return {i: {"X_train": X[idx], "y_train": y[idx]} for i, idx in enumerate(idx_batch)}

    def _serialize_model(self, model) -> str:
        model_bytes = pickle.dumps({"coef": model.coef_, "intercept": model.intercept_})
        return base64.b64encode(model_bytes).decode("utf-8")

    def _deserialize_model(self, model_str: str) -> dict:
        model_bytes = base64.b64decode(model_str.encode("utf-8"))
        return pickle.loads(model_bytes)

    def _model_hash(self, model_str: str) -> str:
        return hashlib.sha256(model_str.encode("utf-8")).hexdigest()

    def _append_round_result(self, *, round_num: int, metrics: dict, avg_local_acc: float, avg_nsds: float,
                             accepted: int, rejected: int, avg_poex_latency_ms: float):
        write_header = not self.results_file.exists()
        with open(self.results_file, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "run_id",
                    "method",
                    "poex_enabled",
                    "poex_threshold",
                    "attack_type",
                    "malicious_ratio",
                    "malicious_clients",
                    "round",
                    "global_accuracy",
                    "global_precision",
                    "global_recall",
                    "global_f1",
                    "avg_local_accuracy",
                    "avg_nsds",
                    "accepted_updates",
                    "rejected_updates",
                    "avg_poex_latency_ms",
                ])
            w.writerow([
                self.run_id,
                self.agg_method,
                int(self.poex_enabled),
                float(self.poex_threshold),
                self.attack_type,
                float(self.malicious_ratio),
                self.malicious_clients,
                int(round_num),
                float(metrics["accuracy"]),
                float(metrics["precision"]),
                float(metrics["recall"]),
                float(metrics["f1"]),
                float(avg_local_acc),
                float(avg_nsds),
                int(accepted),
                int(rejected),
                float(avg_poex_latency_ms),
            ])

    def get_model_payload(self) -> dict:
        return {
            "round": self.current_round,
            "continue": self.current_round < self.max_rounds,
            "model": self._serialize_model(self.global_model),
        }

    def evaluate_global(self) -> dict:
        y_pred = self.global_model.predict(self.X_test)
        return {
            "accuracy": float(np.mean(y_pred == self.y_test)),
            "precision": float(precision_score(self.y_test, y_pred, average='binary', zero_division=0)),
            "recall": float(recall_score(self.y_test, y_pred, average='binary', zero_division=0)),
            "f1": float(f1_score(self.y_test, y_pred, average='binary', zero_division=0)),
        }

    def aggregate_updates(self):
        if not self.client_updates:
            return

        weights_list = []
        shap_list = []
        local_accs = []

        # CRITICAL: Only aggregate accepted updates (PoEx filtering)
        for update in self.client_updates.values():
            if update.get("poex_accepted") is False:
                print(f"[SKIP] Excluding rejected client {update['client_id']} from aggregation")
                continue
            weights_list.append(self._deserialize_model(update["model"]))
            shap_list.append(np.array(update["shap_values"], dtype=float))
            local_accs.append(float(update.get("accuracy", 0.0)))

        # If ALL updates were rejected, skip aggregation but still record the round
        if len(weights_list) == 0:
            print("[WARN] No accepted updates this round - skipping aggregation")
            metrics = self.evaluate_global()  # Eval with current model
            avg_local = 0.0
            avg_nsds = 0.0
        else:
            # NSDS vs mean explanation (local consistency proxy)
            shap_mean = np.mean(shap_list, axis=0)
            eps = 1e-10
            nsds_list = []
            for v in shap_list:
                p = (np.abs(v) + eps) / (np.sum(np.abs(v) + eps))
                q = (np.abs(shap_mean) + eps) / (np.sum(np.abs(shap_mean) + eps))
                nsds_list.append(float(np.sum(p * np.log(p / q))))

            if self.agg_method == "fedavg":
                weights = np.ones(len(weights_list), dtype=float) / max(1, len(weights_list))
            else:
                # If someone sets fedxchain here, use exp(-nsds) weighting as a simple trust proxy.
                scores = np.exp(-np.array(nsds_list, dtype=float))
                weights = scores / max(scores.sum(), eps)

            coef = np.sum([w["coef"] * weights[i] for i, w in enumerate(weights_list)], axis=0)
            # Intercept must remain 1-d array, not 0-d scalar
            intercept = np.sum([w["intercept"] * weights[i] for i, w in enumerate(weights_list)])

            self.global_model.coef_ = coef
            self.global_model.intercept_ = np.array([intercept])  # Ensure shape (1,) not ()

            metrics = self.evaluate_global()
            avg_local = float(np.mean(local_accs)) if local_accs else 0.0
            avg_nsds = float(np.mean(nsds_list)) if nsds_list else 0.0

            # Update chaincode's global explanation after aggregation
            if self.poex_enabled:
                try:
                    self.gateway.update_global_explanation(shap_mean)
                except Exception as e:
                    print(f"! Failed to update global explanation: {e}")

        accepted = int(sum(1 for u in self.client_updates.values() if u.get("poex_accepted") is True))
        rejected = int(sum(1 for u in self.client_updates.values() if u.get("poex_accepted") is False))
        poex_lat = [float(u.get("poex_latency_ms", 0.0)) for u in self.client_updates.values() if u.get("poex_latency_ms") is not None]
        avg_poex_latency_ms = float(np.mean(poex_lat)) if poex_lat else 0.0

        self._append_round_result(
            round_num=self.current_round,
            metrics=metrics,
            avg_local_acc=avg_local,
            avg_nsds=avg_nsds,
            accepted=accepted,
            rejected=rejected,
            avg_poex_latency_ms=avg_poex_latency_ms,
        )

        print(f"[Round {self.current_round}] acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} avg_local={avg_local:.4f} avg_nsds={avg_nsds:.4f} accepted={accepted} rejected={rejected}")

        # next round
        self.current_round += 1
        self.client_updates = {}


aggregator = PoExDistributedAggregator()


@app.get("/health")
def health():
    return jsonify({"status": "ok", "round": aggregator.current_round, "poex_enabled": aggregator.poex_enabled})


@app.post("/register")
def register():
    client_id = int(request.json.get("client_id"))
    aggregator.clients_ready[client_id] = True
    n_samples = len(aggregator.client_data[client_id]["X_train"])
    return jsonify({"status": "registered", "client_id": client_id, "n_samples": n_samples})


@app.get("/get_data/<int:client_id>")
def get_data(client_id: int):
    d = aggregator.client_data[client_id]
    return jsonify(
        {
            "X_train": d["X_train"].tolist(),
            "y_train": d["y_train"].tolist(),
            "X_test": aggregator.X_test.tolist(),
            "y_test": aggregator.y_test.tolist(),
        }
    )


@app.get("/get_model")
def get_model():
    return jsonify(aggregator.get_model_payload())


@app.post("/submit_update")
def submit_update():
    if aggregator.current_round >= aggregator.max_rounds:
        return jsonify({"status": "done", "continue": False, "round": aggregator.current_round})

    payload = request.json
    client_id = int(payload["client_id"])
    model_str = str(payload["model"])
    shap_values = np.array(payload["shap_values"], dtype=float)
    accuracy = float(payload.get("accuracy", 0.0))
    
    print(f"[DEBUG-ENTRY] Received update: client={client_id}, current_round={aggregator.current_round}, shap_shape={shap_values.shape}")

    poex_accepted = True
    poex_trust_after = None
    poex_latency_ms = 0.0

    if aggregator.poex_enabled:
        try:
            t0 = time.time()
            model_hash = aggregator._model_hash(model_str)
            print(f"[DEBUG] Submitting to gateway: client_id={client_id}, round={aggregator.current_round}, hash={model_hash[:8]}...")
            resp = aggregator.gateway.submit_update(
                client_id=client_id,
                round_num=aggregator.current_round,
                model_hash=model_hash,
                explanation=shap_values,
            )
            poex_latency_ms = (time.time() - t0) * 1000.0
            poex_accepted = bool(resp.get("accepted", True))
            poex_trust_after = float(resp.get("trustAfter")) if "trustAfter" in resp else None
        except Exception as e:
            # Fail-closed: if PoEx is enabled, treat gateway failure as rejection
            poex_accepted = False
            poex_trust_after = None
            poex_latency_ms = 0.0
            import traceback
            print(f"! PoEx validation failed (rejecting): {e}")
            print(f"! Full traceback:\n{traceback.format_exc()}")

    # Always record the update (even if rejected) for proper aggregation triggering
    aggregator.client_updates[client_id] = {
        "client_id": client_id,
        "model": model_str,
        "shap_values": shap_values.tolist(),
        "accuracy": accuracy,
        "poex_accepted": poex_accepted,
        "poex_latency_ms": poex_latency_ms,
    }

    if len(aggregator.client_updates) >= aggregator.n_clients:
        aggregator.aggregate_updates()
        status = "rejected" if not poex_accepted else "aggregated"
        return jsonify({
            "status": status,
            "continue": aggregator.current_round < aggregator.max_rounds,
            "round": aggregator.current_round,
            "trust_score": poex_trust_after
        })

    status = "rejected" if not poex_accepted else "waiting"
    return jsonify({
        "status": status,
        "continue": True,
        "round": aggregator.current_round,
        "trust_score": poex_trust_after
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
