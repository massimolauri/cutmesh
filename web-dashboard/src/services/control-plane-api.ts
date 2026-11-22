import {
  AppRole,
  CatalogModel,
  CovariantAuditReport,
  EnrollmentTokenIssued,
  EnrollmentTokenItem,
  GpuWorkerNode,
  ManagedRig,
  OpenAiApiKeyItem,
  SecurityAuditEvent,
  UserProfile
} from "@/types/control-plane";

export const API_BASE =
  process.env.NEXT_PUBLIC_CONTROL_PLANE_URL ||
  (typeof window !== "undefined" && window.location.hostname
    ? `${window.location.protocol}//${window.location.hostname}:${Number(window.location.port) >= 30000 ? "30885" : "8085"
    }`
    : "http://localhost:8085");

const STORAGE_USER_ID_KEY = "cutmesh_active_user_id";
const STORAGE_JWT_TOKEN_KEY = "cutmesh_jwt_token";

export function getStoredJwtToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(STORAGE_JWT_TOKEN_KEY);
}

export function setStoredJwtToken(token: string | null): void {
  if (typeof window === "undefined") return;
  if (token) {
    window.localStorage.setItem(STORAGE_JWT_TOKEN_KEY, token);
  } else {
    window.localStorage.removeItem(STORAGE_JWT_TOKEN_KEY);
  }
}

export function getStoredUserId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(STORAGE_USER_ID_KEY);
}

export function setStoredUserId(userId: string | null): void {
  if (typeof window === "undefined") return;
  if (userId) {
    window.localStorage.setItem(STORAGE_USER_ID_KEY, userId);
  } else {
    window.localStorage.removeItem(STORAGE_USER_ID_KEY);
  }
}

export function clearAuthSession(): void {
  setStoredJwtToken(null);
  setStoredUserId(null);
  if (typeof window !== "undefined") {
    try {
      window.sessionStorage.removeItem("cutmesh_issued_token_cmds_v1");
      window.localStorage.removeItem("cutmesh_issued_token_cmds_v1");
    } catch {
      // ignore storage errors
    }
  }
  if (typeof window !== "undefined") {
    try {
      window.sessionStorage.removeItem("cutmesh_issued_token_cmds_v1");
      window.localStorage.removeItem("cutmesh_issued_token_cmds_v1");
    } catch {
      // ignore storage errors
    }
  }
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...(extra || {}) };
  const jwt = getStoredJwtToken();
  if (jwt) {
    headers["Authorization"] = `Bearer ${jwt}`;
  }
  const uid = getStoredUserId();
  if (uid) {
    headers["X-Cutmesh-User-Id"] = uid;
  }
  return headers;
}

export interface ControlPlaneSnapshot {
  user: UserProfile | null;
  allUsers: UserProfile[];
  workers: GpuWorkerNode[];
  tokens: EnrollmentTokenItem[];
  models: CatalogModel[];
  rigs: ManagedRig[];
  apiKeys: OpenAiApiKeyItem[];
  auditEvents: SecurityAuditEvent[];
  backendOnline: boolean;
  unauthenticated?: boolean;
}

export async function fetchControlPlaneSnapshot(): Promise<ControlPlaneSnapshot> {
  const jwt = getStoredJwtToken();
  if (!jwt) {
    return {
      user: null,
      allUsers: [],
      workers: [],
      tokens: [],
      models: [],
      rigs: [],
      apiKeys: [],
      auditEvents: [],
      backendOnline: true,
      unauthenticated: true
    };
  }

  try {
    const headers = authHeaders();
    const uRes = await fetch(`${API_BASE}/api/v1/auth/me`, { headers });
    if (uRes.status === 401 || uRes.status === 403) {
      clearAuthSession();
      return {
        user: null,
        allUsers: [],
        workers: [],
        tokens: [],
        models: [],
        rigs: [],
        apiKeys: [],
        auditEvents: [],
        backendOnline: true,
        unauthenticated: true
      };
    }

    const user: UserProfile | null = uRes.ok ? await uRes.json() : null;
    if (user?.jwtToken) {
      setStoredJwtToken(user.jwtToken);
    }

    const isAdmin = user?.role === "SUPER_ADMIN";
    const isProviderOrAdmin = isAdmin || user?.role === "HARDWARE_PROVIDER";

    const [usersRes, wRes, tRes, mRes, rRes, kRes, aRes] = await Promise.all([
      isAdmin ? fetch(`${API_BASE}/api/v1/users`, { headers }) : Promise.resolve(null),
      fetch(`${API_BASE}/api/v1/workers`, { headers }),
      isProviderOrAdmin ? fetch(`${API_BASE}/api/v1/enrollment-tokens`, { headers }) : Promise.resolve(null),
      fetch(`${API_BASE}/api/v1/catalog/models`, { headers }),
      fetch(`${API_BASE}/api/v1/rigs?activeOnly=true`, { headers }),
      fetch(`${API_BASE}/api/v1/api-keys`, { headers }),
      isAdmin ? fetch(`${API_BASE}/api/v1/audit/events`, { headers }) : Promise.resolve(null)
    ]);

    return {
      user,
      allUsers: usersRes && usersRes.ok ? await usersRes.json() : [],
      workers: wRes.ok ? await wRes.json() : [],
      tokens: tRes && tRes.ok ? await tRes.json() : [],
      models: mRes.ok ? await mRes.json() : [],
      rigs: rRes.ok ? await rRes.json() : [],
      apiKeys: kRes.ok ? await kRes.json() : [],
      auditEvents: aRes && aRes.ok ? await aRes.json() : [],
      backendOnline: true,
      unauthenticated: false
    };
  } catch {
    return {
      user: null,
      allUsers: [],
      workers: [],
      tokens: [],
      models: [],
      rigs: [],
      apiKeys: [],
      auditEvents: [],
      backendOnline: false
    };
  }
}

export async function loginWithEmailPassword(
  email: string,
  password: string
): Promise<{ user?: UserProfile; error?: string }> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email.trim(), password })
    });
    if (res.ok) {
      const user: UserProfile = await res.json();
      if (user.jwtToken) {
        setStoredJwtToken(user.jwtToken);
      }
      setStoredUserId(user.id);
      return { user };
    }
    const err = await res.json().catch(() => ({}));
    return { error: err.message || "Credenziali non valide" };
  } catch {
    return {
      error: `Impossibile contattare il server (${API_BASE}). Verifica la connessione.`
    };
  }
}

export async function registerUserAccount(payload: {
  email: string;
  password: string;
  displayName: string;
  role: AppRole;
  preferredLocale: "it" | "en";
}): Promise<{ user?: UserProfile; error?: string }> {
  const res = await fetch(`${API_BASE}/api/v1/auth/register`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  if (res.ok) {
    const user: UserProfile = await res.json();
    if (!getStoredJwtToken() && user.jwtToken) {
      setStoredJwtToken(user.jwtToken);
      setStoredUserId(user.id);
    }
    return { user };
  }
  const err = await res.json().catch(() => ({}));
  return { error: err.message || "Registrazione fallita" };
}

export async function loginWithGoogleSso(payload: {
  idToken: string;
  preferredLocale?: "it" | "en";
}): Promise<UserProfile | null> {
  const res = await fetch(`${API_BASE}/api/v1/auth/google`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      idToken: payload.idToken,
      preferredLocale: payload.preferredLocale || "it"
    })
  });
  if (res.ok) {
    const user: UserProfile = await res.json();
    if (user.jwtToken) {
      setStoredJwtToken(user.jwtToken);
    }
    setStoredUserId(user.id);
    return user;
  }
  return null;
}

export interface LiveSwarmPublicStatus {
  backendOnline: boolean;
  rttMs: number;
  onlineWorkersCount: number | null;
  servingRigsCount: number | null;
  totalTflops: number | null;
}

export async function fetchLiveSwarmPublicStatus(): Promise<LiveSwarmPublicStatus> {
  const start = typeof performance !== "undefined" ? performance.now() : Date.now();
  try {
    const jwt = getStoredJwtToken();
    if (jwt) {
      const headers = authHeaders();
      const [wRes, rRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/workers`, { headers }),
        fetch(`${API_BASE}/api/v1/rigs`, { headers })
      ]);
      const rttMs = Math.max(
        1,
        Math.round((typeof performance !== "undefined" ? performance.now() : Date.now()) - start)
      );
      if (wRes.ok && rRes.ok) {
        const workers: GpuWorkerNode[] = await wRes.json();
        const rigs: ManagedRig[] = await rRes.json();
        const onlineWorkers = workers.filter((w) => w.assignmentState !== "OFFLINE_STALE");
        const servingRigs = rigs.filter((r) => r.lifecycleState === "SERVING");
        const totalTflops = onlineWorkers.reduce(
          (acc, w) => acc + (w.effectiveTflops || 0),
          0
        );
        return {
          backendOnline: true,
          rttMs,
          onlineWorkersCount: onlineWorkers.length,
          servingRigsCount: servingRigs.length,
          totalTflops
        };
      }
    }

    // Lightweight public health probe
    const probeRes = await fetch(`${API_BASE}/api/v1/public/status`);
    const rttMs = Math.max(
      1,
      Math.round((typeof performance !== "undefined" ? performance.now() : Date.now()) - start)
    );
    return {
      backendOnline: probeRes.ok,
      rttMs,
      onlineWorkersCount: null,
      servingRigsCount: null,
      totalTflops: null
    };
  } catch {
    return {
      backendOnline: false,
      rttMs: 0,
      onlineWorkersCount: null,
      servingRigsCount: null,
      totalTflops: null
    };
  }
}

export async function loginWithApiToken(
  rawToken: string,
  preferredLocale: "it" | "en" = "it"
): Promise<{ user?: UserProfile; apiKey?: string; error?: string }> {
  const token = rawToken.trim();
  if (!token) {
    return {
      error:
        preferredLocale === "it"
          ? "Inserisci un API Token (sk-cutmesh-...) o un token JWT valido."
          : "Please enter a valid API Token (sk-cutmesh-...) or JWT token."
    };
  }

  try {
    // Case 1: OpenAI-compatible Cutmesh API Key
    if (token.startsWith("sk-cutmesh-") || token.startsWith("sk-synapse-")) {
      const authRes = await fetch(`${API_BASE}/api/v1/auth/api-key`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apiKey: token })
      });
      if (!authRes.ok) {
        return {
          error:
            preferredLocale === "it"
              ? "Chiave API sk-cutmesh non valida o revocata."
              : "Invalid or revoked sk-cutmesh API key."
        };
      }
      const user: UserProfile = await authRes.json();
      if (user.jwtToken) {
        setStoredJwtToken(user.jwtToken);
      }
      setStoredUserId(user.id);
      return { user, apiKey: token };
    }

    // Case 2: RFC 7519 JWT Bearer Token (eyJ...)
    const meRes = await fetch(`${API_BASE}/api/v1/auth/me`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (meRes.ok) {
      const user: UserProfile = await meRes.json();
      setStoredJwtToken(user.jwtToken || token);
      setStoredUserId(user.id);
      return { user };
    }
    return {
      error:
        preferredLocale === "it"
          ? "Token JWT non valido o scaduto."
          : "Invalid or expired JWT Bearer token."
    };
  } catch {
    return {
      error:
        preferredLocale === "it"
          ? "Control Plane non raggiungibile durante la verifica del token."
          : "Control Plane unreachable during token verification."
    };
  }
}

export async function updateActiveUserRole(role: AppRole): Promise<UserProfile | null> {
  const res = await fetch(`${API_BASE}/api/v1/auth/role`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ role })
  });
  if (res.ok) {
    const updated: UserProfile = await res.json();
    if (updated.jwtToken) {
      setStoredJwtToken(updated.jwtToken);
    }
    return updated;
  }
  return null;
}

export async function updateTargetUserRole(
  targetUserId: string,
  role: AppRole
): Promise<UserProfile | null> {
  const res = await fetch(`${API_BASE}/api/v1/users/${targetUserId}/role`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ role })
  });
  return res.ok ? await res.json() : null;
}

export async function deleteTargetUser(targetUserId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/v1/users/${targetUserId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  return res.ok;
}

export async function issueEnrollmentToken(
  label: string,
  defaultPoolMode: "SHARED_POOL" | "PRIVATE_SOLO"
): Promise<EnrollmentTokenIssued | null> {
  const res = await fetch(`${API_BASE}/api/v1/enrollment-tokens`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ label, defaultPoolMode })
  });
  return res.ok ? await res.json() : null;
}

export async function revokeEnrollmentToken(tokenId: string): Promise<void> {
  await fetch(`${API_BASE}/api/v1/enrollment-tokens/${tokenId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
}

export async function triggerWatchdogFault(workerId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/v1/workers/${workerId}/test-watchdog`, {
    method: "POST",
    headers: authHeaders()
  });
  return res.ok;
}

export async function disconnectWorkerNode(workerId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/v1/workers/${workerId}/disconnect`, {
    method: "POST",
    headers: authHeaders()
  });
  return res.ok;
}

export async function deleteWorkerNode(workerId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/v1/workers/${workerId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  return res.ok;
}

export async function registerCatalogModel(payload: {
  modelId: string;
  displayName: string;
  huggingfaceRepo: string;
  baseGgufPath: string;
  securityProfile: "FULL_COVARIANT_SHIELD" | "SHARD_ONLY_NO_COVARIANT";
  orchestrationMode: "AUTO_PILOT" | "MANUAL_ONLY";
  priorityOrder?: number;
}): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/v1/catalog/models`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      ...payload,
      minActiveRigs: 1,
      maxActiveRigs: 4,
      priorityOrder: payload.priorityOrder ?? 1
    })
  });
  return res.ok;
}

export async function deleteCatalogModel(modelId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/v1/catalog/models/${encodeURIComponent(modelId)}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  return res.ok;
}

export async function toggleModelOrchestrationMode(model: CatalogModel): Promise<string> {
  const nextMode = model.orchestrationMode === "AUTO_PILOT" ? "MANUAL_ONLY" : "AUTO_PILOT";
  await fetch(`${API_BASE}/api/v1/catalog/models/${model.modelId}/orchestration`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      orchestrationMode: nextMode,
      minActiveRigs: model.minActiveRigs,
      maxActiveRigs: model.maxActiveRigs,
      securityProfile: model.securityProfile,
      priorityOrder: model.priorityOrder || 1
    })
  });
  return nextMode;
}

export async function updateModelPriorityOrder(
  model: CatalogModel,
  priorityOrder: number
): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/v1/catalog/models/${model.modelId}/orchestration`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      orchestrationMode: model.orchestrationMode,
      minActiveRigs: model.minActiveRigs,
      maxActiveRigs: model.maxActiveRigs,
      securityProfile: model.securityProfile,
      priorityOrder
    })
  });
  return res.ok;
}

export async function compilePhysicalCovariantGguf(
  modelId: string,
  secretSeed: string,
  kappaLambda: number
): Promise<any | null> {
  const res = await fetch(
    `${API_BASE}/api/v1/catalog/models/${modelId}/compile-covariant`,
    {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ secretSeed, kappaLambda })
    }
  );
  return res.ok ? await res.json() : null;
}

export async function createManualRig(
  rigName: string,
  modelId: string,
  workerNodeIds: string[]
): Promise<{ ok: boolean; errorMessage?: string }> {
  const res = await fetch(`${API_BASE}/api/v1/rigs/manual`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ rigName, modelId, workerNodeIds })
  });
  if (res.ok) return { ok: true };
  const err = await res.json().catch(() => ({}));
  return { ok: false, errorMessage: err.message || "VRAM insufficiente" };
}

export async function dissolveManagedRig(rigId: string): Promise<void> {
  await fetch(`${API_BASE}/api/v1/rigs/${rigId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
}

export async function runLiveCovariantAudit(
  dimension: number,
  modelId: string = "qwen3.8-27b"
): Promise<CovariantAuditReport | null> {
  const res = await fetch(`${API_BASE}/api/v1/audit/run-covariant`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      dimension,
      seed: 42,
      secretSeed: "cutmesh-covariant-session-secret-2026",
      modelId
    })
  });
  return res.ok ? await res.json() : null;
}

export async function createOpenAiApiKey(
  name: string,
  rateLimitRpm: number
): Promise<{ rawApiKeySecret: string; keyPrefix: string } | null> {
  const res = await fetch(`${API_BASE}/api/v1/api-keys`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ name, rateLimitRpm })
  });
  return res.ok ? await res.json() : null;
}

export async function revokeOpenAiApiKey(keyId: string): Promise<void> {
  await fetch(`${API_BASE}/api/v1/api-keys/${keyId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
}
