import { useState, useRef, useEffect, useCallback } from 'react';
import { Send, Bot, Loader2, Paperclip, FileText, X, ChevronDown, ChevronRight, ExternalLink } from 'lucide-react';
import { sendMessage, ChatWebSocket, uploadCV } from '../services/api';
import { cn } from '../lib/utils';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/* ═══════════════════ Markdown renderer ═══════════════════ */

function ChatMarkdown({ content }: { content: string }) {
    return (
        <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
                /* ── Headings ── */
                h1: ({ children }) => (
                    <h1 className="text-base font-bold text-slate-800 mt-3 mb-1.5 first:mt-0">{children}</h1>
                ),
                h2: ({ children }) => (
                    <h2 className="text-sm font-bold text-slate-800 mt-2.5 mb-1 first:mt-0">{children}</h2>
                ),
                h3: ({ children }) => (
                    <h3 className="text-sm font-semibold text-slate-700 mt-2 mb-1 first:mt-0 flex items-center gap-1.5">{children}</h3>
                ),
                /* ── Paragraph ── */
                p: ({ children }) => (
                    <p className="text-sm leading-relaxed mb-1.5 last:mb-0">{children}</p>
                ),
                /* ── Bold / Italic ── */
                strong: ({ children }) => (
                    <strong className="font-semibold text-slate-800">{children}</strong>
                ),
                em: ({ children }) => (
                    <em className="text-slate-500 not-italic">{children}</em>
                ),
                /* ── Links ── */
                a: ({ href, children }) => (
                    <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-700 underline decoration-blue-200 hover:decoration-blue-400 underline-offset-2 transition-colors inline-flex items-center gap-0.5"
                    >
                        {children}
                        <ExternalLink size={10} className="shrink-0 opacity-50" />
                    </a>
                ),
                /* ── Lists ── */
                ul: ({ children }) => (
                    <ul className="space-y-0.5 mb-1.5 ml-1">{children}</ul>
                ),
                ol: ({ children }) => (
                    <ol className="space-y-0.5 mb-1.5 ml-1 list-decimal list-inside">{children}</ol>
                ),
                li: ({ children }) => (
                    <li className="text-sm leading-relaxed flex items-start gap-1.5">
                        <span className="text-blue-400 mt-1 shrink-0">•</span>
                        <span className="flex-1">{children}</span>
                    </li>
                ),
                /* ── Table ── */
                table: ({ children }) => (
                    <div className="my-2 rounded-lg border border-slate-200 overflow-hidden overflow-x-auto">
                        <table className="w-full text-xs">{children}</table>
                    </div>
                ),
                thead: ({ children }) => (
                    <thead className="bg-slate-50 border-b border-slate-200">{children}</thead>
                ),
                tbody: ({ children }) => (
                    <tbody className="divide-y divide-slate-100">{children}</tbody>
                ),
                tr: ({ children }) => (
                    <tr className="hover:bg-blue-50/30 transition-colors">{children}</tr>
                ),
                th: ({ children }) => (
                    <th className="px-3 py-2 text-left text-[11px] font-semibold text-slate-500 uppercase tracking-wider">{children}</th>
                ),
                td: ({ children }) => (
                    <td className="px-3 py-2 text-slate-600 align-top">{children}</td>
                ),
                /* ── Code ── */
                code: ({ className, children, ...props }) => {
                    const isInline = !className;
                    return isInline ? (
                        <code className="bg-slate-100 text-indigo-600 px-1.5 py-0.5 rounded text-xs font-mono" {...props}>{children}</code>
                    ) : (
                        <code className="block bg-slate-900 text-slate-100 p-3 rounded-lg text-xs font-mono overflow-x-auto my-1.5" {...props}>{children}</code>
                    );
                },
                pre: ({ children }) => (
                    <pre className="my-1.5">{children}</pre>
                ),
                /* ── Blockquote ── */
                blockquote: ({ children }) => (
                    <blockquote className="border-l-3 border-blue-300 pl-3 py-0.5 my-1.5 text-slate-500 italic">{children}</blockquote>
                ),
                /* ── Horizontal rule ── */
                hr: () => (
                    <hr className="my-2 border-slate-200" />
                ),
            }}
        >
            {content}
        </ReactMarkdown>
    );
}

interface Message {
    role: 'user' | 'assistant';
    content: string;
    reasoningLog?: string[];
}

export default function Chat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [wsConnected, setWsConnected] = useState(false);
    const [uploadedCVs, setUploadedCVs] = useState<string[]>([]);
    const [liveLog, setLiveLog] = useState<string[]>([]);
    const [expandedLogs, setExpandedLogs] = useState<Set<number>>(new Set());

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<ChatWebSocket | null>(null);
    const contextRef = useRef<any>({});
    const fileInputRef = useRef<HTMLInputElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(scrollToBottom, [messages, liveLog]);

    // WebSocket with real-time log streaming
    useEffect(() => {
        const ws = new ChatWebSocket();
        wsRef.current = ws;

        ws.onMessage((data) => {
            if (data.type === 'log') {
                // Real-time log step — append to live log
                setLiveLog(prev => [...prev, data.step]);
            } else {
                // Final response (type === 'response' or legacy)
                const log = data.reasoning_log || [];
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: data.response,
                    reasoningLog: log.length > 0 ? log : undefined,
                }]);
                contextRef.current = data.context || contextRef.current;
                setLiveLog([]);
                setIsLoading(false);
            }
        });

        ws.onError(() => setWsConnected(false));

        ws.connect()
            .then(() => setWsConnected(true))
            .catch(() => setWsConnected(false));

        return () => { ws.disconnect(); };
    }, []);

    const handleSend = useCallback(async () => {
        if (!input.trim() || isLoading) return;

        setMessages(prev => [...prev, { role: 'user', content: input }]);
        const currentInput = input;
        setInput('');
        setIsLoading(true);
        setLiveLog([]);

        if (wsRef.current?.isConnected) {
            wsRef.current.send(currentInput, contextRef.current);
        } else {
            try {
                setLiveLog(['⏳ Processing...']);
                const data = await sendMessage(currentInput, contextRef.current);
                const log = data.reasoning_log || [];
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: data.response,
                    reasoningLog: log.length > 0 ? log : undefined,
                }]);
                contextRef.current = data.context || contextRef.current;
            } catch {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: '❌ Cannot reach the server. Is the backend running?'
                }]);
            } finally {
                setLiveLog([]);
                setIsLoading(false);
            }
        }
    }, [input, isLoading]);

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        if (!file.name.toLowerCase().endsWith('.pdf')) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: '❌ Only PDF files are supported.'
            }]);
            return;
        }

        setMessages(prev => [...prev, { role: 'user', content: `📎 ${file.name}` }]);
        setIsLoading(true);
        setLiveLog(['📄 Parsing PDF...', '🔍 Extracting text...']);

        try {
            const r = await uploadCV(file);

            let content = `✅ **CV Analyzed**\n`;
            if (r.candidate_name) content += `👤 ${r.candidate_name}\n`;
            if (r.job_title) content += `💼 ${r.job_title}\n`;
            content += `📄 ${r.filename}`;
            if (r.pages) content += ` · ${r.pages} page${r.pages > 1 ? 's' : ''}`;
            if (r.word_count) content += ` · ${r.word_count} words`;
            content += '\n';
            if (r.total_experience) content += `📅 ${r.total_experience} years experience\n`;
            if (r.skills?.length) content += `🛠️ ${r.skills.join(', ')}\n`;
            if (r.summary) content += `\n${r.summary}`;

            setMessages(prev => [...prev, {
                role: 'assistant',
                content,
                reasoningLog: [
                    `📄 File: ${r.filename}`,
                    `🔧 Extraction: ${r.extraction_method || 'auto'}`,
                    `📊 ${r.word_count || 0} words extracted`,
                    `🛠️ ${r.skills?.length || 0} skills found`,
                ],
            }]);
            setUploadedCVs(prev => [...prev, r.filename]);
            contextRef.current = {
                ...contextRef.current,
                lastUploadedCV: r.filename,
                candidateData: r,
                current_cv_text: r.text || '',
            };
        } catch {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: '❌ Failed to analyze CV. Check that the backend is running.'
            }]);
        } finally {
            setLiveLog([]);
            setIsLoading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const toggleLog = (idx: number) => {
        setExpandedLogs(prev => {
            const next = new Set(prev);
            next.has(idx) ? next.delete(idx) : next.add(idx);
            return next;
        });
    };

    // ── Render ──

    return (
        <div className="flex flex-col h-[calc(100vh-4rem)] max-w-3xl mx-auto">

            {/* ── Messages area ── */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3">

                {/* Empty state */}
                {messages.length === 0 && (
                    <div className="flex flex-col items-center justify-center h-full text-center select-none">
                        <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center mb-3 shadow-lg shadow-indigo-200">
                            <Bot size={28} className="text-white" />
                        </div>
                        <h1 className="text-xl font-semibold text-slate-800">ATIA-HR</h1>
                        <p className="text-sm text-slate-400 mb-6">Intelligent Recruitment Assistant</p>

                        <div className="grid grid-cols-2 gap-2 w-full max-w-xs">
                            {[
                                { icon: '🔍', label: 'Search Python remote jobs', value: 'Search for Python remote jobs' },
                                { icon: '📄', label: 'Upload a CV', action: () => fileInputRef.current?.click() },
                                { icon: '🏆', label: 'Rank candidates', value: 'Rank the uploaded candidates' },
                                { icon: '📝', label: 'Generate an offer', value: 'Draft an offer for Senior Developer' },
                            ].map((item, i) => (
                                <button
                                    key={i}
                                    onClick={() => item.action ? item.action() : setInput(item.value!)}
                                    className="flex items-center gap-2 p-2.5 text-xs text-left bg-white border border-slate-200 rounded-lg hover:border-indigo-300 hover:shadow-sm transition-all"
                                >
                                    <span>{item.icon}</span>
                                    <span className="text-slate-600">{item.label}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Message list */}
                {messages.map((msg, idx) => (
                    <div key={idx} className={cn('flex gap-2.5', msg.role === 'user' ? 'justify-end' : 'justify-start')}>

                        {/* Bot avatar */}
                        {msg.role === 'assistant' && (
                            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-0.5">
                                <Bot size={14} className="text-white" />
                            </div>
                        )}

                        <div className={cn('flex flex-col', msg.role === 'user' ? 'items-end' : 'items-start', 'max-w-[75%]')}>
                            {/* Bubble */}
                            <div className={cn(
                                'px-3 py-2 rounded-2xl text-sm leading-relaxed',
                                msg.role === 'user'
                                    ? 'bg-indigo-600 text-white rounded-br-md whitespace-pre-wrap'
                                    : 'bg-white border border-slate-200 text-slate-700 rounded-bl-md shadow-sm'
                            )}>
                                {msg.role === 'user' ? msg.content : <ChatMarkdown content={msg.content} />}
                            </div>

                            {/* Collapsible reasoning log */}
                            {msg.role === 'assistant' && msg.reasoningLog && msg.reasoningLog.length > 0 && (
                                <button
                                    onClick={() => toggleLog(idx)}
                                    className="flex items-center gap-1 mt-1 text-[10px] text-slate-400 hover:text-indigo-500 transition-colors"
                                >
                                    {expandedLogs.has(idx) ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
                                    <span>Process log ({msg.reasoningLog.length} steps)</span>
                                </button>
                            )}
                            {msg.role === 'assistant' && msg.reasoningLog && expandedLogs.has(idx) && (
                                <div className="mt-1 w-full rounded-lg bg-slate-900 border border-slate-700/50 p-2 text-[11px] font-mono leading-relaxed" style={{ opacity: 0.7 }}>
                                    {msg.reasoningLog.map((step, i) => (
                                        <div key={i} className="text-slate-300 py-0.5">
                                            {step}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                ))}

                {/* ── Live processing indicator + real-time log ── */}
                {isLoading && (
                    <div className="flex gap-2.5 justify-start">
                        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-0.5">
                            <Loader2 size={14} className="text-white animate-spin" />
                        </div>
                        <div className="flex flex-col items-start max-w-[75%]">
                            <div className="px-3 py-2 rounded-2xl rounded-bl-md bg-white border border-slate-200 shadow-sm">
                                <span className="text-sm text-slate-400 italic">Thinking...</span>
                            </div>

                            {/* Real-time reasoning log */}
                            {liveLog.length > 0 && (
                                <div className="mt-1 w-full rounded-lg bg-slate-900 border border-slate-700/50 p-2 text-[11px] font-mono leading-relaxed overflow-hidden" style={{ opacity: 0.7 }}>
                                    {liveLog.map((step, i) => (
                                        <div
                                            key={i}
                                            className={cn(
                                                'text-slate-300 py-0.5 transition-all duration-300',
                                                i === liveLog.length - 1 && 'text-green-400 font-medium'
                                            )}
                                        >
                                            {step}
                                        </div>
                                    ))}
                                    <div className="flex items-center gap-1 mt-1 text-slate-500">
                                        <Loader2 size={9} className="animate-spin" />
                                        <span className="text-[10px]">processing...</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* ── Input bar ── */}
            <div className="p-3 border-t border-slate-100 bg-white/80 backdrop-blur-sm">

                {/* CV badges */}
                {uploadedCVs.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-2">
                        {uploadedCVs.map((name, i) => (
                            <span key={i} className="inline-flex items-center gap-1 text-[10px] bg-indigo-50 text-indigo-600 pl-1.5 pr-1 py-0.5 rounded-full">
                                <FileText size={9} />
                                {name}
                                <button
                                    onClick={() => setUploadedCVs(prev => prev.filter((_, idx) => idx !== i))}
                                    className="hover:text-red-500 ml-0.5"
                                >
                                    <X size={9} />
                                </button>
                            </span>
                        ))}
                    </div>
                )}

                <div className="flex items-center gap-2">
                    <input type="file" ref={fileInputRef} onChange={handleFileUpload} accept=".pdf" className="hidden" />
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isLoading}
                        className="p-2 text-slate-400 hover:text-indigo-500 hover:bg-indigo-50 rounded-lg transition-colors disabled:opacity-40"
                        title="Upload CV (PDF)"
                    >
                        <Paperclip size={18} />
                    </button>
                    <input
                        className="flex-1 px-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent placeholder:text-slate-400"
                        placeholder="Ask anything..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
                        disabled={isLoading}
                    />
                    <button
                        onClick={handleSend}
                        disabled={isLoading || !input.trim()}
                        className="p-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl disabled:opacity-40 transition-colors"
                    >
                        <Send size={18} />
                    </button>
                </div>

                {/* Status bar */}
                <div className="flex items-center justify-between mt-1.5 px-1">
                    <span className="text-[10px] text-slate-400">ATIA-HR</span>
                    <span className={cn(
                        'text-[10px] flex items-center gap-1',
                        wsConnected ? 'text-emerald-500' : 'text-slate-400'
                    )}>
                        <span className={cn(
                            'w-1.5 h-1.5 rounded-full',
                            wsConnected ? 'bg-emerald-500' : 'bg-slate-300'
                        )} />
                        {wsConnected ? 'Connected' : 'HTTP'}
                    </span>
                </div>
            </div>
        </div>
    );
}
