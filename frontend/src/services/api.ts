import axios from 'axios';

const API_BASE = 'http://localhost:8001';
const WS_BASE = 'ws://localhost:8001';

const api = axios.create({
    baseURL: API_BASE,
});

export const checkHealth = async () => {
    const response = await api.get('/health');
    return response.data;
}

// ── HTTP chat (fallback) ──────────────────────────────────────
export const sendMessage = async (message: string, context?: any) => {
    const response = await api.post('/chat', { message, context });
    return response.data;
};

// ── WebSocket chat (real-time) ────────────────────────────────
export class ChatWebSocket {
    private ws: WebSocket | null = null;
    private onMessageCallback: ((data: any) => void) | null = null;
    private onErrorCallback: ((error: any) => void) | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;

    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(`${WS_BASE}/ws/chat`);

                this.ws.onopen = () => {
                    this.reconnectAttempts = 0;
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.onMessageCallback?.(data);
                    } catch (e) {
                        console.error('WS parse error:', e);
                    }
                };

                this.ws.onerror = (error) => {
                    this.onErrorCallback?.(error);
                    reject(error);
                };

                this.ws.onclose = () => {
                    if (this.reconnectAttempts < this.maxReconnectAttempts) {
                        this.reconnectAttempts++;
                        setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
                    }
                };
            } catch (e) {
                reject(e);
            }
        });
    }

    send(message: string, context?: any) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ message, context }));
        }
    }

    onMessage(callback: (data: any) => void) {
        this.onMessageCallback = callback;
    }

    onError(callback: (error: any) => void) {
        this.onErrorCallback = callback;
    }

    get isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    disconnect() {
        this.maxReconnectAttempts = 0; // prevent reconnect
        this.ws?.close();
        this.ws = null;
    }
}

// ── Job search ────────────────────────────────────────────────
export interface SearchFilters {
    query: string;
    sources?: string[];
    max_results?: number;
    location?: string;
    remote_only?: boolean;
    experience_level?: string;
}

export const searchJobs = async (query: string, filters?: Partial<SearchFilters>) => {
    const response = await api.post('/search', {
        query,
        ...filters,
    });
    return response.data;
};

// ── CV-based job recommendations ──────────────────────────────
export interface RecommendFilters {
    sources?: string[];
    max_results?: number;
    location?: string;
    remote_only?: boolean;
}

export const recommendJobs = async (filters?: Partial<RecommendFilters>) => {
    const response = await api.post('/search/recommend', filters || {});
    return response.data;
};

// ── CV analysis ───────────────────────────────────────────────
export const uploadCV = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/candidates/analyze', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

// ── Candidate ranking ─────────────────────────────────────────
export const rankCandidates = async (jobDescription: string, filenames?: string[]) => {
    const response = await api.post('/candidates/rank', {
        job_description: jobDescription,
        filenames,
    });
    return response.data;
};

export const getCachedCVs = async () => {
    const response = await api.get('/candidates/cached');
    return response.data;
};

// ── Hiring / Offer generation ─────────────────────────────────
export const generateOffer = async (data: any) => {
    const response = await api.post('/hiring/offer', data);
    return response.data;
};

export default api;
