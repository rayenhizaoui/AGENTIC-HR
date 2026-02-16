import { useState, useEffect, useCallback } from 'react';
import {
    Upload,
    FileText,
    CheckCircle,
    Loader2,
    Users,
    Briefcase,
    Search,
    Trash2,
    ChevronDown,
    ChevronUp,
    Award,
    BarChart3,
    Sparkles,
    Eye,
    GraduationCap,
    Clock,
    Shield,
    AlertTriangle,
    TrendingUp,
    Star,
    XCircle,
    RefreshCw,
    FolderOpen,
    Code2,
    Layers,
    BadgeCheck,
    Hash,
    Calendar,
} from 'lucide-react';
import { uploadCV, getCachedCVs, rankCandidates } from '../services/api';

/* ═══════════════════ Types ═══════════════════ */

interface CachedCV {
    filename: string;
    skills_count: number;
    summary_preview: string;
}

interface AnalysisResult {
    filename: string;
    text: string;
    summary: string;
    skills_data: {
        skills: string[];
        skill_categories?: Record<string, string[]>;
        experience_years: number;
        education: { degree: string; field: string; institution: string; year?: number }[];
        certifications?: string[];
        projects_count?: number;
        candidate_name?: string;
        job_title?: string;
        semantic_matches?: { skill: string; score: number }[];
        synonym_expansions?: string[];
        date_ranges?: [string, string][];
        note?: string;
    };
    skills: string[];
    pages: number;
    word_count: number;
    extraction_method: string;
    ocr_confidence?: number;
    pii_redactions?: Record<string, number>;
    candidate_name?: string;
    job_title?: string;
    total_experience?: number;
}

/* ═══════════════════ Markdown-light renderer ═══════════════════ */

function RenderSummary({ text }: { text: string }) {
    if (!text) return null;
    const lines = text.split('\n');
    return (
        <div className="text-sm leading-relaxed text-slate-600 space-y-1.5">
            {lines.map((line, i) => {
                if (!line.trim()) return <div key={i} className="h-1" />;
                // Render bold (**text**) and italic (*text*)
                const rendered = line
                    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-slate-800 font-semibold">$1</strong>')
                    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em class="text-slate-500">$1</em>');
                const isBullet = line.trimStart().startsWith('- ');
                if (isBullet) {
                    const content = rendered.replace(/^\s*-\s*/, '');
                    return (
                        <div key={i} className="flex items-start gap-2 pl-2">
                            <span className="text-blue-400 mt-1.5 shrink-0">•</span>
                            <span dangerouslySetInnerHTML={{ __html: content }} />
                        </div>
                    );
                }
                return <p key={i} dangerouslySetInnerHTML={{ __html: rendered }} />;
            })}
        </div>
    );
}

interface RankingResult {
    candidate: string;
    score: number;
    semantic_similarity: number;
    skill_match: number;
    matched_skills: string[];
    missing_skills: string[];
    summary?: string;
    llm_used: boolean;
}

type Tab = 'upload' | 'library' | 'rank';

const TABS: { key: Tab; label: string; icon: React.ElementType }[] = [
    { key: 'upload', label: 'Upload & Analyze', icon: Upload },
    { key: 'library', label: 'CV Library', icon: Users },
    { key: 'rank', label: 'Rank for Job', icon: Award },
];

/* ═══════════════════ Score helpers ═══════════════════ */

function scoreColor(score: number) {
    if (score >= 70) return 'text-emerald-600';
    if (score >= 45) return 'text-amber-600';
    return 'text-red-500';
}

function scoreBg(score: number) {
    if (score >= 70) return 'bg-emerald-50 border-emerald-200';
    if (score >= 45) return 'bg-amber-50 border-amber-200';
    return 'bg-red-50 border-red-200';
}

function scoreBarColor(score: number) {
    if (score >= 70) return 'bg-emerald-500';
    if (score >= 45) return 'bg-amber-500';
    return 'bg-red-400';
}

function medalIcon(rank: number) {
    if (rank === 0) return '🥇';
    if (rank === 1) return '🥈';
    if (rank === 2) return '🥉';
    return `#${rank + 1}`;
}

/* ═══════════════════ Main Component ═══════════════════ */

export default function Analyze() {
    const [tab, setTab] = useState<Tab>('upload');

    /* ── Upload state ── */
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [dragOver, setDragOver] = useState(false);

    /* ── Library state ── */
    const [cachedCVs, setCachedCVs] = useState<CachedCV[]>([]);
    const [libLoading, setLibLoading] = useState(false);
    const [selectedCVs, setSelectedCVs] = useState<Set<string>>(new Set());

    /* ── Ranking state ── */
    const [jobDesc, setJobDesc] = useState('');
    const [rankings, setRankings] = useState<RankingResult[]>([]);
    const [rankLoading, setRankLoading] = useState(false);
    const [rankError, setRankError] = useState('');
    const [expandedRank, setExpandedRank] = useState<number | null>(null);

    /* ── Load cached CVs on tab switch ── */
    const loadCachedCVs = useCallback(async () => {
        setLibLoading(true);
        try {
            const data = await getCachedCVs();
            setCachedCVs(data.cached_cvs || []);
        } catch {
            setCachedCVs([]);
        } finally {
            setLibLoading(false);
        }
    }, []);

    useEffect(() => {
        if (tab === 'library' || tab === 'rank') loadCachedCVs();
    }, [tab, loadCachedCVs]);

    /* ── Handlers ── */
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) {
            setFile(e.target.files[0]);
            setError('');
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files[0];
        if (f && (f.name.endsWith('.pdf') || f.name.endsWith('.docx'))) {
            setFile(f);
            setError('');
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;
        setLoading(true);
        setError('');
        setResult(null);
        try {
            const data = await uploadCV(file);
            setResult(data);
            // Refresh library
            loadCachedCVs();
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to analyze CV. Make sure the backend is running.');
        } finally {
            setLoading(false);
        }
    };

    const handleRank = async () => {
        if (!jobDesc.trim()) return;
        setRankLoading(true);
        setRankError('');
        setRankings([]);
        setExpandedRank(null);
        try {
            const filenames = selectedCVs.size > 0 ? Array.from(selectedCVs) : undefined;
            const data = await rankCandidates(jobDesc, filenames);
            setRankings(data.rankings || []);
        } catch (err: any) {
            setRankError(err.response?.data?.detail || 'Ranking failed. Upload CVs first.');
        } finally {
            setRankLoading(false);
        }
    };

    const toggleCV = (filename: string) => {
        setSelectedCVs(prev => {
            const next = new Set(prev);
            if (next.has(filename)) next.delete(filename);
            else next.add(filename);
            return next;
        });
    };

    const toggleAll = () => {
        if (selectedCVs.size === cachedCVs.length) setSelectedCVs(new Set());
        else setSelectedCVs(new Set(cachedCVs.map(c => c.filename)));
    };

    /* ═══════════════════ RENDER ═══════════════════ */
    return (
        <div className="max-w-6xl mx-auto">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
                    <BarChart3 className="text-blue-600" size={28} />
                    CV Analysis & Ranking
                </h1>
                <p className="text-slate-500 text-sm mt-1">
                    Upload CVs, build your candidate library, and rank them for any job position
                </p>
            </div>

            {/* Tab bar */}
            <div className="flex gap-1 bg-slate-100 p-1 rounded-xl mb-6">
                {TABS.map(t => {
                    const Icon = t.icon;
                    const isActive = tab === t.key;
                    return (
                        <button
                            key={t.key}
                            onClick={() => setTab(t.key)}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium flex-1 justify-center transition-all ${
                                isActive
                                    ? 'bg-white text-blue-700 shadow-sm'
                                    : 'text-slate-500 hover:text-slate-700 hover:bg-white/50'
                            }`}
                        >
                            <Icon size={16} />
                            {t.label}
                            {t.key === 'library' && cachedCVs.length > 0 && (
                                <span className={`ml-1 text-xs px-1.5 py-0.5 rounded-full ${
                                    isActive ? 'bg-blue-100 text-blue-700' : 'bg-slate-200 text-slate-600'
                                }`}>
                                    {cachedCVs.length}
                                </span>
                            )}
                        </button>
                    );
                })}
            </div>

            {/* ────────── TAB 1: Upload & Analyze ────────── */}
            {tab === 'upload' && (
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                    {/* Left: Upload panel */}
                    <div className="lg:col-span-2 space-y-4">
                        <div
                            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                            onDragLeave={() => setDragOver(false)}
                            onDrop={handleDrop}
                            className={`bg-white p-8 rounded-xl border-2 border-dashed text-center transition-all cursor-pointer ${
                                dragOver
                                    ? 'border-blue-400 bg-blue-50'
                                    : file
                                        ? 'border-emerald-300 bg-emerald-50/30'
                                        : 'border-slate-200 hover:border-blue-300 hover:bg-slate-50'
                            }`}
                        >
                            <input
                                type="file"
                                id="cv-upload"
                                className="hidden"
                                accept=".pdf,.docx"
                                onChange={handleFileChange}
                            />
                            <label htmlFor="cv-upload" className="cursor-pointer flex flex-col items-center gap-3">
                                <div className={`w-14 h-14 rounded-full flex items-center justify-center ${
                                    file ? 'bg-emerald-100 text-emerald-600' : 'bg-blue-50 text-blue-600'
                                }`}>
                                    {file ? <CheckCircle size={28} /> : <Upload size={28} />}
                                </div>
                                {file ? (
                                    <div>
                                        <p className="font-medium text-slate-800 text-sm">{file.name}</p>
                                        <p className="text-xs text-slate-400 mt-0.5">
                                            {(file.size / 1024).toFixed(0)} KB — Click to change
                                        </p>
                                    </div>
                                ) : (
                                    <div>
                                        <p className="text-sm">
                                            <span className="font-semibold text-blue-600">Click to upload</span>
                                            <span className="text-slate-500"> or drag & drop</span>
                                        </p>
                                        <p className="text-xs text-slate-400 mt-1">PDF or DOCX — Max 5MB</p>
                                    </div>
                                )}
                            </label>
                        </div>

                        <button
                            onClick={handleAnalyze}
                            disabled={!file || loading}
                            className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-3 rounded-xl font-medium hover:bg-blue-700 disabled:opacity-40 transition-colors shadow-sm"
                        >
                            {loading ? (
                                <><Loader2 size={18} className="animate-spin" /> Analyzing with AI...</>
                            ) : (
                                <><Sparkles size={18} /> Analyze CV</>
                            )}
                        </button>

                        {error && (
                            <div className="p-3 bg-red-50 text-red-700 rounded-xl border border-red-200 text-sm flex items-start gap-2">
                                <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                                {error}
                            </div>
                        )}

                        {/* Quick stats when result available */}
                        {result && (
                            <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-3">
                                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Document Info</h3>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    {[
                                        { icon: FileText, label: 'Pages', value: result.pages },
                                        { icon: FileText, label: 'Words', value: result.word_count?.toLocaleString() },
                                        { icon: Shield, label: 'Method', value: result.extraction_method },
                                        ...(result.ocr_confidence ? [{ icon: Eye, label: 'OCR Conf.', value: `${result.ocr_confidence}%` }] : []),
                                    ].map(item => (
                                        <div key={item.label} className="flex items-center gap-2 bg-slate-50 rounded-lg px-3 py-2">
                                            <item.icon size={13} className="text-slate-400" />
                                            <div>
                                                <p className="text-[10px] text-slate-400 uppercase">{item.label}</p>
                                                <p className="font-medium text-slate-700">{item.value}</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                {result.pii_redactions && Object.keys(result.pii_redactions).length > 0 && (
                                    <div className="flex items-center gap-2 text-xs text-amber-700 bg-amber-50 rounded-lg px-3 py-2 border border-amber-200">
                                        <Shield size={14} />
                                        PII detected & redacted: {Object.entries(result.pii_redactions).map(([k, v]) => `${v} ${k}`).join(', ')}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Right: Analysis result */}
                    <div className="lg:col-span-3">
                        {result ? (
                            <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
                                {/* Header */}
                                <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 rounded-full bg-white/20 flex items-center justify-center text-white font-bold text-xl">
                                            {(result.candidate_name || result.filename)?.[0]?.toUpperCase() || '?'}
                                        </div>
                                        <div>
                                            <h2 className="text-lg font-semibold text-white">
                                                {result.candidate_name || result.filename}
                                            </h2>
                                            <p className="text-blue-100 text-sm">
                                                {result.job_title || 'Candidate'}
                                                {result.total_experience ? ` — ${result.total_experience}+ years exp.` : ''}
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                <div className="p-6 space-y-5">
                                    {/* Summary — rendered with markdown-light */}
                                    <div>
                                        <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                                            <Sparkles size={14} className="text-blue-500" /> Professional Summary
                                        </h3>
                                        <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                                            <RenderSummary text={typeof result.summary === 'string' ? result.summary : JSON.stringify(result.summary)} />
                                        </div>
                                    </div>

                                    {/* Quick stats row */}
                                    <div className="grid grid-cols-4 gap-3">
                                        <div className="bg-blue-50 rounded-xl p-3 border border-blue-100 text-center">
                                            <Clock size={16} className="text-blue-500 mx-auto mb-1" />
                                            <p className="text-xl font-bold text-slate-800">{result.skills_data?.experience_years || 0}</p>
                                            <p className="text-[10px] text-slate-500 uppercase font-semibold">Years Exp.</p>
                                        </div>
                                        <div className="bg-indigo-50 rounded-xl p-3 border border-indigo-100 text-center">
                                            <GraduationCap size={16} className="text-indigo-500 mx-auto mb-1" />
                                            <p className="text-xl font-bold text-slate-800">{result.skills_data?.education?.length || 0}</p>
                                            <p className="text-[10px] text-slate-500 uppercase font-semibold">Degrees</p>
                                        </div>
                                        <div className="bg-emerald-50 rounded-xl p-3 border border-emerald-100 text-center">
                                            <Award size={16} className="text-emerald-500 mx-auto mb-1" />
                                            <p className="text-xl font-bold text-slate-800">{result.skills_data?.skills?.length || 0}</p>
                                            <p className="text-[10px] text-slate-500 uppercase font-semibold">Skills</p>
                                        </div>
                                        <div className="bg-amber-50 rounded-xl p-3 border border-amber-100 text-center">
                                            <FolderOpen size={16} className="text-amber-500 mx-auto mb-1" />
                                            <p className="text-xl font-bold text-slate-800">{result.skills_data?.projects_count || 0}</p>
                                            <p className="text-[10px] text-slate-500 uppercase font-semibold">Projects</p>
                                        </div>
                                    </div>

                                    {/* Skills by category */}
                                    {result.skills_data?.skill_categories && Object.keys(result.skills_data.skill_categories).length > 0 ? (
                                        <div>
                                            <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                                                <Layers size={14} className="text-blue-500" /> Skills by Category
                                            </h3>
                                            <div className="space-y-3">
                                                {Object.entries(result.skills_data.skill_categories).map(([cat, skills]) => (
                                                    <div key={cat}>
                                                        <p className="text-xs font-semibold text-slate-500 mb-1.5 flex items-center gap-1">
                                                            <Code2 size={12} className="text-slate-400" />
                                                            {cat}
                                                            <span className="text-slate-300 font-normal">({skills.length})</span>
                                                        </p>
                                                        <div className="flex flex-wrap gap-1.5">
                                                            {skills.map((skill: string, i: number) => (
                                                                <span key={i} className="px-2.5 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-medium border border-blue-100">
                                                                    {skill}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                            {/* Uncategorized skills (from Skills section parsing) */}
                                            {(() => {
                                                const catSkills = new Set(
                                                    Object.values(result.skills_data.skill_categories || {}).flat().map((s: string) => s.toLowerCase())
                                                );
                                                const extra = (result.skills_data?.skills || []).filter(s => !catSkills.has(s.toLowerCase()));
                                                if (extra.length === 0) return null;
                                                return (
                                                    <div className="mt-3">
                                                        <p className="text-xs font-semibold text-slate-500 mb-1.5 flex items-center gap-1">
                                                            <Hash size={12} className="text-slate-400" />
                                                            Other Skills
                                                            <span className="text-slate-300 font-normal">({extra.length})</span>
                                                        </p>
                                                        <div className="flex flex-wrap gap-1.5">
                                                            {extra.map((skill: string, i: number) => (
                                                                <span key={i} className="px-2.5 py-1 bg-slate-50 text-slate-600 rounded-full text-xs font-medium border border-slate-200">
                                                                    {skill}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </div>
                                                );
                                            })()}
                                            {result.skills_data?.semantic_matches && result.skills_data.semantic_matches.length > 0 && (
                                                <p className="text-[11px] text-slate-400 mt-2">
                                                    + {result.skills_data.semantic_matches.length} additional skills discovered via semantic analysis
                                                </p>
                                            )}
                                        </div>
                                    ) : (
                                        /* Flat skills list fallback */
                                        <div>
                                            <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                                                <Award size={14} className="text-blue-500" /> Skills ({result.skills_data?.skills?.length || 0})
                                            </h3>
                                            <div className="flex flex-wrap gap-1.5">
                                                {result.skills_data?.skills?.map((skill: string, i: number) => (
                                                    <span key={i} className="px-2.5 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-medium border border-blue-100">
                                                        {skill}
                                                    </span>
                                                ))}
                                            </div>
                                            {result.skills_data?.semantic_matches && result.skills_data.semantic_matches.length > 0 && (
                                                <p className="text-[11px] text-slate-400 mt-2">
                                                    + {result.skills_data.semantic_matches.length} additional skills discovered via semantic analysis
                                                </p>
                                            )}
                                        </div>
                                    )}

                                    {/* Education details */}
                                    {result.skills_data?.education && result.skills_data.education.length > 0 && (
                                        <div>
                                            <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                                                <GraduationCap size={14} className="text-blue-500" /> Education
                                            </h3>
                                            <div className="space-y-2">
                                                {result.skills_data.education.map((edu, i) => (
                                                    <div key={i} className="flex items-start gap-3 text-sm bg-slate-50 rounded-lg px-4 py-3 border border-slate-100">
                                                        <div className="w-8 h-8 rounded-full bg-indigo-100 flex items-center justify-center shrink-0 mt-0.5">
                                                            <GraduationCap size={14} className="text-indigo-500" />
                                                        </div>
                                                        <div className="flex-1">
                                                            <p className="font-semibold text-slate-700">
                                                                {edu.degree}{edu.field ? ` in ${edu.field}` : ''}
                                                            </p>
                                                            <div className="flex items-center gap-2 mt-0.5">
                                                                {edu.institution && (
                                                                    <p className="text-xs text-slate-400">{edu.institution}</p>
                                                                )}
                                                                {edu.year && (
                                                                    <span className="text-xs text-slate-300">•</span>
                                                                )}
                                                                {edu.year && (
                                                                    <p className="text-xs text-slate-400">{edu.year}</p>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Certifications */}
                                    {result.skills_data?.certifications && result.skills_data.certifications.length > 0 && (
                                        <div>
                                            <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                                                <BadgeCheck size={14} className="text-blue-500" /> Certifications
                                            </h3>
                                            <div className="flex flex-wrap gap-1.5">
                                                {result.skills_data.certifications.map((cert, i) => (
                                                    <span key={i} className="px-2.5 py-1 bg-emerald-50 text-emerald-700 rounded-full text-xs font-medium border border-emerald-200">
                                                        {cert}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Experience timeline info */}
                                    {result.skills_data?.date_ranges && result.skills_data.date_ranges.length > 0 && (
                                        <div>
                                            <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide mb-2 flex items-center gap-1.5">
                                                <Calendar size={14} className="text-blue-500" /> Career Timeline
                                            </h3>
                                            <div className="flex flex-wrap gap-2">
                                                {result.skills_data.date_ranges.map(([start, end], i) => (
                                                    <span key={i} className="px-2.5 py-1 bg-slate-50 text-slate-600 rounded-lg text-xs font-medium border border-slate-200">
                                                        {new Date(start).toLocaleDateString('en', { month: 'short', year: 'numeric' })}
                                                        {' → '}
                                                        {new Date(end).toLocaleDateString('en', { month: 'short', year: 'numeric' })}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Raw CV text — expandable */}
                                    {result.text && (
                                        <details className="group">
                                            <summary className="cursor-pointer text-sm font-semibold text-slate-500 uppercase tracking-wide flex items-center gap-1.5 hover:text-blue-600 transition-colors">
                                                <FileText size={14} className="text-slate-400 group-hover:text-blue-500" />
                                                Raw CV Text
                                                <ChevronDown size={14} className="ml-auto group-open:rotate-180 transition-transform" />
                                            </summary>
                                            <div className="mt-2 bg-slate-50 p-4 rounded-lg border border-slate-100 max-h-64 overflow-y-auto">
                                                <pre className="text-xs text-slate-600 whitespace-pre-wrap font-mono leading-relaxed">
                                                    {result.text}
                                                </pre>
                                            </div>
                                        </details>
                                    )}

                                    {/* Action: go to ranking */}
                                    <button
                                        onClick={() => setTab('rank')}
                                        className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-2.5 rounded-xl text-sm font-medium hover:bg-blue-700 transition-colors shadow-sm"
                                    >
                                        <Award size={16} /> Rank this candidate for a job
                                    </button>
                                </div>
                            </div>
                        ) : (
                            /* Empty state */
                            <div className="bg-white border border-slate-200 rounded-xl shadow-sm flex flex-col items-center justify-center py-20 text-center">
                                <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                                    <FileText size={28} className="text-slate-300" />
                                </div>
                                <p className="text-slate-500 font-medium">No CV analyzed yet</p>
                                <p className="text-slate-400 text-sm mt-1">Upload a CV on the left to see the analysis results here</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ────────── TAB 2: CV Library ────────── */}
            {tab === 'library' && (
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <p className="text-sm text-slate-500">
                            {cachedCVs.length} CV{cachedCVs.length !== 1 ? 's' : ''} in your session library
                        </p>
                        <button
                            onClick={loadCachedCVs}
                            className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-blue-600 transition-colors"
                        >
                            <RefreshCw size={14} /> Refresh
                        </button>
                    </div>

                    {libLoading ? (
                        <div className="flex items-center justify-center py-16">
                            <Loader2 size={24} className="animate-spin text-blue-500" />
                        </div>
                    ) : cachedCVs.length === 0 ? (
                        <div className="bg-white border border-slate-200 rounded-xl flex flex-col items-center justify-center py-16 text-center">
                            <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                                <Users size={28} className="text-slate-300" />
                            </div>
                            <p className="text-slate-500 font-medium">No CVs uploaded yet</p>
                            <p className="text-slate-400 text-sm mt-1 max-w-sm">
                                Go to the Upload tab to analyze CVs. They'll appear here for ranking.
                            </p>
                            <button
                                onClick={() => setTab('upload')}
                                className="mt-4 px-4 py-2 bg-blue-600 text-white text-sm rounded-lg font-medium hover:bg-blue-700 transition-colors"
                            >
                                Upload a CV
                            </button>
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                            {cachedCVs.map((cv, i) => (
                                <div
                                    key={cv.filename}
                                    className="bg-white border border-slate-200 rounded-xl p-4 hover:shadow-md transition-shadow group"
                                >
                                    <div className="flex items-start gap-3">
                                        <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 font-bold text-sm shrink-0">
                                            {cv.filename[0]?.toUpperCase() || '?'}
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="font-medium text-slate-800 text-sm truncate" title={cv.filename}>
                                                {cv.filename}
                                            </p>
                                            <div className="flex items-center gap-2 mt-1">
                                                <span className="text-xs text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full font-medium">
                                                    {cv.skills_count} skills
                                                </span>
                                            </div>
                                            {cv.summary_preview && (
                                                <p className="text-xs text-slate-400 mt-2 line-clamp-2">
                                                    {cv.summary_preview}
                                                </p>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* ────────── TAB 3: Rank for Job ────────── */}
            {tab === 'rank' && (
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                    {/* Left: Job description + CV selection */}
                    <div className="lg:col-span-2 space-y-4">
                        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
                            <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                                <Briefcase size={16} className="text-blue-500" />
                                Job Description
                            </h3>
                            <textarea
                                value={jobDesc}
                                onChange={e => setJobDesc(e.target.value)}
                                placeholder="Paste the full job description here...&#10;&#10;Example: We are looking for a Senior Python Developer with 5+ years of experience in FastAPI, machine learning, and cloud services (AWS/GCP). The ideal candidate should have experience with LLMs, RAG pipelines, and Docker/Kubernetes..."
                                className="w-full h-48 px-4 py-3 border border-slate-200 rounded-xl text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none placeholder:text-slate-300"
                            />

                            {/* CV selection */}
                            <div>
                                <div className="flex items-center justify-between mb-2">
                                    <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                                        Select CVs to rank ({selectedCVs.size}/{cachedCVs.length})
                                    </h4>
                                    {cachedCVs.length > 0 && (
                                        <button onClick={toggleAll} className="text-xs text-blue-600 hover:text-blue-700 font-medium">
                                            {selectedCVs.size === cachedCVs.length ? 'Deselect all' : 'Select all'}
                                        </button>
                                    )}
                                </div>

                                {cachedCVs.length === 0 ? (
                                    <div className="text-sm text-slate-400 bg-slate-50 rounded-lg p-3 text-center">
                                        No CVs available. Upload CVs first.
                                    </div>
                                ) : (
                                    <div className="space-y-1.5 max-h-40 overflow-y-auto">
                                        {cachedCVs.map(cv => (
                                            <label
                                                key={cv.filename}
                                                className={`flex items-center gap-2.5 px-3 py-2 rounded-lg cursor-pointer text-sm transition-colors ${
                                                    selectedCVs.has(cv.filename)
                                                        ? 'bg-blue-50 border border-blue-200'
                                                        : 'bg-slate-50 border border-transparent hover:bg-slate-100'
                                                }`}
                                            >
                                                <input
                                                    type="checkbox"
                                                    checked={selectedCVs.has(cv.filename)}
                                                    onChange={() => toggleCV(cv.filename)}
                                                    className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                                                />
                                                <span className="truncate text-slate-700 font-medium">{cv.filename}</span>
                                                <span className="text-xs text-slate-400 ml-auto shrink-0">{cv.skills_count} skills</span>
                                            </label>
                                        ))}
                                    </div>
                                )}
                                <p className="text-[11px] text-slate-400 mt-2">
                                    Leave empty to rank all CVs
                                </p>
                            </div>

                            <button
                                onClick={handleRank}
                                disabled={rankLoading || !jobDesc.trim() || cachedCVs.length === 0}
                                className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-3 rounded-xl font-medium hover:bg-blue-700 disabled:opacity-40 transition-colors shadow-sm"
                            >
                                {rankLoading ? (
                                    <><Loader2 size={18} className="animate-spin" /> Ranking with AI...</>
                                ) : (
                                    <><TrendingUp size={18} /> Rank Candidates</>
                                )}
                            </button>

                            {rankError && (
                                <div className="p-3 bg-red-50 text-red-700 rounded-xl border border-red-200 text-sm flex items-start gap-2">
                                    <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                                    {rankError}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Right: Ranking results */}
                    <div className="lg:col-span-3 space-y-3">
                        {rankings.length > 0 ? (
                            <>
                                <div className="flex items-center justify-between mb-2">
                                    <h3 className="text-sm font-semibold text-slate-600">
                                        {rankings.length} Candidate{rankings.length !== 1 ? 's' : ''} Ranked
                                    </h3>
                                    <span className="text-xs text-slate-400">
                                        {rankings[0]?.llm_used ? 'Mistral LLM + Embeddings' : 'Embeddings only'}
                                    </span>
                                </div>

                                {rankings.map((r, i) => (
                                    <div
                                        key={r.candidate}
                                        className={`bg-white border rounded-xl overflow-hidden transition-all ${
                                            i === 0 ? 'border-emerald-200 shadow-md' : 'border-slate-200'
                                        }`}
                                    >
                                        {/* Card header */}
                                        <div
                                            className="flex items-center gap-3 px-5 py-4 cursor-pointer hover:bg-slate-50/50 transition-colors"
                                            onClick={() => setExpandedRank(expandedRank === i ? null : i)}
                                        >
                                            <span className="text-xl shrink-0 w-8 text-center">{medalIcon(i)}</span>
                                            <div className="flex-1 min-w-0">
                                                <p className="font-semibold text-slate-800 text-sm truncate">{r.candidate}</p>
                                                {r.summary && (
                                                    <p className="text-xs text-slate-400 truncate mt-0.5">{r.summary.slice(0, 80)}...</p>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-3 shrink-0">
                                                {/* Score badge */}
                                                <div className={`px-3 py-1.5 rounded-lg border font-bold text-lg ${scoreBg(r.score)} ${scoreColor(r.score)}`}>
                                                    {r.score}%
                                                </div>
                                                {expandedRank === i ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
                                            </div>
                                        </div>

                                        {/* Score bars (always visible) */}
                                        <div className="px-5 pb-3">
                                            <div className="flex items-center gap-4 text-xs">
                                                <div className="flex-1">
                                                    <div className="flex justify-between mb-0.5">
                                                        <span className="text-slate-400">Semantic</span>
                                                        <span className="font-medium text-slate-600">{r.semantic_similarity}%</span>
                                                    </div>
                                                    <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                                        <div className={`h-full rounded-full transition-all ${scoreBarColor(r.semantic_similarity)}`} style={{ width: `${r.semantic_similarity}%` }} />
                                                    </div>
                                                </div>
                                                <div className="flex-1">
                                                    <div className="flex justify-between mb-0.5">
                                                        <span className="text-slate-400">Skill Match</span>
                                                        <span className="font-medium text-slate-600">{r.skill_match}%</span>
                                                    </div>
                                                    <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                                        <div className={`h-full rounded-full transition-all ${scoreBarColor(r.skill_match)}`} style={{ width: `${r.skill_match}%` }} />
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Expanded detail */}
                                        {expandedRank === i && (
                                            <div className="px-5 pb-4 pt-1 border-t border-slate-100 space-y-3">
                                                {r.matched_skills.length > 0 && (
                                                    <div>
                                                        <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide mb-1.5 flex items-center gap-1">
                                                            <CheckCircle size={12} /> Matched Skills ({r.matched_skills.length})
                                                        </p>
                                                        <div className="flex flex-wrap gap-1.5">
                                                            {r.matched_skills.map(s => (
                                                                <span key={s} className="px-2 py-0.5 bg-emerald-50 text-emerald-700 rounded-full text-xs font-medium border border-emerald-200">
                                                                    {s}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                                {r.missing_skills.length > 0 && (
                                                    <div>
                                                        <p className="text-xs font-semibold text-red-500 uppercase tracking-wide mb-1.5 flex items-center gap-1">
                                                            <XCircle size={12} /> Missing Skills ({r.missing_skills.length})
                                                        </p>
                                                        <div className="flex flex-wrap gap-1.5">
                                                            {r.missing_skills.map(s => (
                                                                <span key={s} className="px-2 py-0.5 bg-red-50 text-red-600 rounded-full text-xs font-medium border border-red-200">
                                                                    {s}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </>
                        ) : (
                            <div className="bg-white border border-slate-200 rounded-xl flex flex-col items-center justify-center py-20 text-center">
                                <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                                    <Award size={28} className="text-slate-300" />
                                </div>
                                <p className="text-slate-500 font-medium">No rankings yet</p>
                                <p className="text-slate-400 text-sm mt-1 max-w-sm">
                                    Paste a job description on the left and click "Rank Candidates" to see who's the best fit
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
