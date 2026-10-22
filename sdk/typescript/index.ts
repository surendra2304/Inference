/**
 * TypeScript SDK Client for Inference Multi-Agent Intelligence Platform.
 */

export interface IntelligenceRequest {
  request_id: string;
  task_type: string;
  goal: string;
  context?: Record<string, any>;
  evidence?: Array<{ claim: string; trust_label: string }>;
  mode?: 'fast' | 'review' | 'debate';
}

export interface IntelligenceResponse {
  request_id: string;
  decision: string;
  confidence: number;
  summary: string;
  key_evidence: string[];
  unresolved_disagreements: string[];
  recommended_actions: Array<{ action: string; priority: string }>;
}

export class AIUniverseClient {
  private baseUrl: string;
  private apiKey?: string;

  constructor(options?: { baseUrl?: string; apiKey?: string }) {
    this.baseUrl = (options?.baseUrl || 'http://localhost:8000').replace(/\/$/, '');
    this.apiKey = options?.apiKey;
  }

  private async request<T>(endpoint: string, options: RequestInit): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {})
    };

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: { ...headers, ...options.headers }
    });

    if (!response.ok) {
      throw new Error(`Inference API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  public async queryNexusIntelligence(request: IntelligenceRequest): Promise<IntelligenceResponse> {
    return this.request<IntelligenceResponse>('/v1/nexus/intelligence', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }

  public async enhanceStatisticalForecast(payload: {
    request_id: string;
    statistical_forecast: Record<string, any>;
    target_context?: Record<string, any>;
    contextual_factors?: string[];
    question?: string;
  }): Promise<Record<string, any>> {
    return this.request('/v1/futuris/enhance', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  public async queryIntelXResearch(payload: {
    request_id: string;
    role: 'planner' | 'extractor' | 'verifier' | 'analyst' | 'critic' | 'synthesizer';
    context: Record<string, any>;
    evidence_with_spans?: Array<Record<string, any>>;
    constraints?: Record<string, any>;
  }): Promise<Record<string, any>> {
    return this.request('/v1/intelx/research', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  public async querySentinelAnalysis(payload: {
    request_id: string;
    analysis_type: string;
    target_context: Record<string, any>;
    findings?: Array<Record<string, any>>;
    threat_intel?: Record<string, any>;
  }): Promise<Record<string, any>> {
    return this.request('/v1/sentinel/analyze', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  public async generateCode(payload: {
    file_type: string;
    filename: string;
    context: Record<string, any>;
    requirements?: string[];
  }): Promise<{ code: string; confidence: number; generation_path: string }> {
    return this.request('/v1/forge/generate-code', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  public async reportOutcome(payload: {
    consumer: string;
    request_id: string;
    outcome: 'success' | 'partial' | 'failure';
    detail?: string;
    measured_metrics?: Record<string, any>;
  }): Promise<{ status: string; request_id: string }> {
    return this.request('/v1/analytics/outcome', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }
}
