package io.cutmesh.controlplane.orchestration;

/**
 * Deterministic Finite State Machine (FSM) governing the physical lifecycle of a closed GPU Rig
 * from initial layer partitioning to live `llama-server` readiness (`/health 200 OK`).
 */
public enum RigRuntimePhaseState {
    STEP_1_LAYERS_ALLOCATED(
        1,
        "LAYERS_ALLOCATED",
        "FASE 1/4 · LAYER ALLOCATI SULLE GPU",
        "Topologia Water-Filling calcolata e range layer assegnati ai Worker GPU"
    ),
    STEP_2_PREWARMING_SHARDS(
        2,
        "PREWARMING_SHARDS",
        "FASE 2/4 · PRE-WARM SHARD GGUF",
        "Download autonomo e verifica hash FNV-1a dei layer nella cache ggml-rpc dei Worker"
    ),
    STEP_3_STARTING_LLAMA_VRAM(
        3,
        "LOADING_VRAM",
        "FASE 3/4 · AVVIO LLAMA-SERVER & CARICAMENTO VRAM",
        "Tunnel Iroh QUIC RPC attivi e caricamento tensori nelle VRAM delle GPU in corso (/health 503)"
    ),
    STEP_4_OPERATIONAL_READY(
        4,
        "OPERATIONAL",
        "FASE 4/4 · OPERATIVO & COMUNICANTE (200 OK)",
        "Motore llama-server avviato, pesi in VRAM e pronto a servire richieste OpenAI /v1/chat/completions"
    ),
    ERROR_DEGRADED(
        0,
        "ERROR_DEGRADED",
        "STATO DEGRADATO / IN RIPRISTINO",
        "Il processo runtime ha segnalato un'anomalia; Self-Healing attivo"
    );

    private final int stepNumber;
    private final String code;
    private final String titleIt;
    private final String descriptionIt;

    RigRuntimePhaseState(int stepNumber, String code, String titleIt, String descriptionIt) {
        this.stepNumber = stepNumber;
        this.code = code;
        this.titleIt = titleIt;
        this.descriptionIt = descriptionIt;
    }

    public int getStepNumber() {
        return stepNumber;
    }

    public String getCode() {
        return code;
    }

    public String getTitleIt() {
        return titleIt;
    }

    public String getDescriptionIt() {
        return descriptionIt;
    }

    /**
     * Enforces valid forward state transitions in the Rig Runtime FSM.
     */
    public boolean canTransitionTo(RigRuntimePhaseState next) {
        if (next == null || next == this) {
            return false;
        }
        if (next == ERROR_DEGRADED || this == ERROR_DEGRADED) {
            return true;
        }
        return next.stepNumber >= this.stepNumber;
    }
}

