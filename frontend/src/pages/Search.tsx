import { useState } from 'react';
import {
    Search as SearchIcon,
    MapPin,
    Briefcase,
    ExternalLink,
    SlidersHorizontal,
    Wifi,
    Star,
    X,
    Loader2,
    Globe,
    Sparkles,
    FileText,
    TrendingUp,
    Zap,
} from 'lucide-react';
import { searchJobs, recommendJobs, getCachedCVs } from '../services/api';

interface Job {
    title: string;
    company: string;
    location: string;
    link: string;
    source?: string;
    relevance?: number;
    compatibility?: number;
    description?: string;
    published?: string;
}

type TabMode = 'search' | 'recommend';

const ALL_SOURCES = [
    'Remote OK', 'We Work Remotely', 'Arbeitnow', 'The Muse',
    'Remotive', 'Himalayas', 'Emploi.tn',
];

const QUICK_SEARCHES = [
    { label: 'Python Developer', icon: '🐍' },
    { label: 'React Frontend', icon: '⚛️' },
    { label: 'Data Scientist', icon: '📊' },
    { label: 'DevOps Engineer', icon: '⚙️' },
    { label: 'AI / ML Engineer', icon: '🤖' },
    { label: 'Full Stack', icon: '🔗' },
    { label: 'Mobile Developer', icon: '📱' },
    { label: 'Cloud Architect', icon: '☁️' },
    { label: 'Tunisia Software', icon: '🇹🇳' },
];

const SOURCE_COLORS: Record<string, string> = {
    'Remote OK': 'bg-green-100 text-green-700',
    'We Work Remotely': 'bg-purple-100 text-purple-700',
    'Arbeitnow': 'bg-orange-100 text-orange-700',
    'The Muse': 'bg-blue-100 text-blue-700',
    'Remotive': 'bg-teal-100 text-teal-700',
    'Himalayas': 'bg-pink-100 text-pink-700',
    'Emploi.tn': 'bg-red-100 text-red-700',
    'TanitJobs': 'bg-red-100 text-red-700',
};

export default function Search() {
    const [tabMode, setTabMode] = useState<TabMode>('search');
    const [query, setQuery] = useState('');
    const [location, setLocation] = useState('');
    const [remoteOnly, setRemoteOnly] = useState(false);
    const [experienceLevel, setExperienceLevel] = useState('any');
    const [selectedSources, setSelectedSources] = useState<string[]>([...ALL_SOURCES]);
    const [maxResults, setMaxResults] = useState(20);
    const [showFilters, setShowFilters] = useState(false);

    const [results, setResults] = useState<Job[]>([]);
    const [loading, setLoading] = useState(false);
    const [searched, setSearched] = useState(false);
    const [meta, setMeta] = useState<{ keywords_used?: string[]; total_found?: number }>({});

    // Recommend state
    const [recResults, setRecResults] = useState<Job[]>([]);
    const [recLoading, setRecLoading] = useState(false);
    const [recSearched, setRecSearched] = useState(false);
    const [recMeta, setRecMeta] = useState<{
        cv_filename?: string;
        cv_skills?: string[];
        search_query_used?: string;
        total_found?: number;
    }>({});
    const [recError, setRecError] = useState('');

    const handleSearch = async (searchQuery?: string) => {
        const q = searchQuery || query;
        if (!q.trim()) return;

        if (searchQuery) setQuery(searchQuery);
        setLoading(true);
        setSearched(true);

        try {
            const data = await searchJobs(q, {
                sources: selectedSources,
                location: location || undefined,
                remote_only: remoteOnly,
                experience_level: experienceLevel !== 'any' ? experienceLevel : undefined,
                max_results: maxResults,
            });
            setResults(data.jobs || []);
            setMeta({ keywords_used: data.keywords_used, total_found: data.total_found });
        } catch (error) {
            console.error(error);
            setResults([]);
        } finally {
            setLoading(false);
        }
    };

    const toggleSource = (source: string) => {
        setSelectedSources((prev) =>
            prev.includes(source) ? prev.filter((s) => s !== source) : [...prev, source]
        );
    };

    const clearFilters = () => {
        setLocation('');
        setRemoteOnly(false);
        setExperienceLevel('any');
        setSelectedSources([...ALL_SOURCES]);
        setMaxResults(20);
    };

    const handleRecommend = async () => {
        setRecLoading(true);
        setRecSearched(true);
        setRecError('');
        try {
            // Check if a CV exists first
            const cached = await getCachedCVs();
            if (!cached.candidates || cached.candidates.length === 0) {
                setRecError('No CV uploaded yet. Go to the Analyze page and upload a CV first.');
                setRecResults([]);
                setRecLoading(false);
                return;
            }
            const data = await recommendJobs({
                sources: selectedSources,
                max_results: maxResults,
                location: location || undefined,
                remote_only: remoteOnly,
            });
            setRecResults(data.jobs || []);
            setRecMeta({
                cv_filename: data.cv_filename,
                cv_skills: data.cv_skills,
                search_query_used: data.search_query_used,
                total_found: data.total_found,
            });
        } catch (error: any) {
            const msg = error?.response?.data?.detail || 'Failed to get recommendations.';
            setRecError(msg);
            setRecResults([]);
        } finally {
            setRecLoading(false);
        }
    };

    const compatibilityColor = (score: number) => {
        if (score >= 70) return 'text-emerald-700 bg-emerald-50 border-emerald-200';
        if (score >= 50) return 'text-blue-700 bg-blue-50 border-blue-200';
        if (score >= 30) return 'text-yellow-700 bg-yellow-50 border-yellow-200';
        return 'text-slate-500 bg-slate-50 border-slate-200';
    };

    const hasActiveFilters =
        location || remoteOnly || experienceLevel !== 'any' || selectedSources.length !== ALL_SOURCES.length || maxResults !== 20;

    const relevanceColor = (score: number) => {
        if (score >= 60) return 'text-green-600 bg-green-50';
        if (score >= 30) return 'text-yellow-600 bg-yellow-50';
        return 'text-slate-500 bg-slate-50';
    };

    return (
        <div className="max-w-5xl mx-auto">
            {/* ── Header ── */}
            <div className="mb-6">
                <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
                    <Globe className="text-blue-600" size={28} />
                    Job Search
                </h1>
                <p className="text-slate-500 text-sm mt-1">
                    Search across 7+ job boards or get AI-matched recommendations from your CV
                </p>

                {/* Tab Switcher */}
                <div className="flex gap-1 mt-4 bg-slate-100 p-1 rounded-xl w-fit">
                    <button
                        onClick={() => setTabMode('search')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-1.5 ${
                            tabMode === 'search'
                                ? 'bg-white text-blue-700 shadow-sm'
                                : 'text-slate-500 hover:text-slate-700'
                        }`}
                    >
                        <SearchIcon size={16} /> Search Jobs
                    </button>
                    <button
                        onClick={() => setTabMode('recommend')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-1.5 ${
                            tabMode === 'recommend'
                                ? 'bg-white text-emerald-700 shadow-sm'
                                : 'text-slate-500 hover:text-slate-700'
                        }`}
                    >
                        <Zap size={16} /> CV Match
                    </button>
                </div>
            </div>

            {/* ── Search Bar ── */}
            {tabMode === 'search' && (
            <>
            <form
                onSubmit={(e) => {
                    e.preventDefault();
                    handleSearch();
                }}
                className="space-y-3"
            >
                <div className="flex gap-2">
                    <div className="relative flex-1">
                        <SearchIcon className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
                        <input
                            className="w-full pl-11 pr-4 py-3 bg-white border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm text-sm"
                            placeholder="Role, skills, or keywords — e.g. Python Developer, React, AI..."
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                        />
                    </div>
                    <button
                        type="button"
                        onClick={() => setShowFilters(!showFilters)}
                        className={`px-3.5 py-3 rounded-xl border transition-colors flex items-center gap-1.5 text-sm font-medium ${
                            showFilters || hasActiveFilters
                                ? 'bg-blue-50 border-blue-200 text-blue-700'
                                : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50'
                        }`}
                    >
                        <SlidersHorizontal size={18} />
                        Filters
                        {hasActiveFilters && (
                            <span className="w-2 h-2 rounded-full bg-blue-600" />
                        )}
                    </button>
                    <button
                        type="submit"
                        disabled={loading || !query.trim()}
                        className="bg-blue-600 text-white px-6 py-3 rounded-xl font-medium hover:bg-blue-700 disabled:opacity-40 transition-colors flex items-center gap-2 text-sm"
                    >
                        {loading ? <Loader2 size={18} className="animate-spin" /> : <SearchIcon size={18} />}
                        {loading ? 'Searching...' : 'Search'}
                    </button>
                </div>

                {/* ── Filter Panel ── */}
                {showFilters && (
                    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
                        <div className="flex items-center justify-between">
                            <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
                                <Sparkles size={16} className="text-blue-500" />
                                Smart Filters
                            </h3>
                            {hasActiveFilters && (
                                <button
                                    type="button"
                                    onClick={clearFilters}
                                    className="text-xs text-slate-500 hover:text-red-500 flex items-center gap-1 transition-colors"
                                >
                                    <X size={14} /> Reset all
                                </button>
                            )}
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            {/* Location */}
                            <div>
                                <label className="block text-xs font-medium text-slate-500 mb-1.5">Location</label>
                                <div className="relative">
                                    <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                                    <input
                                        className="w-full pl-9 pr-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                        placeholder="City, country..."
                                        value={location}
                                        onChange={(e) => setLocation(e.target.value)}
                                    />
                                </div>
                            </div>

                            {/* Experience Level */}
                            <div>
                                <label className="block text-xs font-medium text-slate-500 mb-1.5">Experience</label>
                                <select
                                    className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                    value={experienceLevel}
                                    onChange={(e) => setExperienceLevel(e.target.value)}
                                >
                                    <option value="any">All Levels</option>
                                    <option value="junior">Junior / Entry</option>
                                    <option value="mid">Mid-Level</option>
                                    <option value="senior">Senior / Lead</option>
                                </select>
                            </div>

                            {/* Max Results */}
                            <div>
                                <label className="block text-xs font-medium text-slate-500 mb-1.5">
                                    Max results: {maxResults}
                                </label>
                                <input
                                    type="range"
                                    min={5}
                                    max={50}
                                    step={5}
                                    value={maxResults}
                                    onChange={(e) => setMaxResults(Number(e.target.value))}
                                    className="w-full accent-blue-600 mt-1"
                                />
                            </div>
                        </div>

                        {/* Remote Toggle */}
                        <div className="flex items-center gap-3">
                            <button
                                type="button"
                                onClick={() => setRemoteOnly(!remoteOnly)}
                                className={`relative w-10 h-5 rounded-full transition-colors ${
                                    remoteOnly ? 'bg-blue-600' : 'bg-slate-300'
                                }`}
                            >
                                <span
                                    className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
                                        remoteOnly ? 'translate-x-5' : ''
                                    }`}
                                />
                            </button>
                            <span className="text-sm text-slate-600 flex items-center gap-1.5">
                                <Wifi size={15} className={remoteOnly ? 'text-blue-600' : 'text-slate-400'} />
                                Remote only
                            </span>
                        </div>

                        {/* Sources */}
                        <div>
                            <label className="block text-xs font-medium text-slate-500 mb-2">Job Boards</label>
                            <div className="flex flex-wrap gap-2">
                                {ALL_SOURCES.map((source) => {
                                    const active = selectedSources.includes(source);
                                    return (
                                        <button
                                            key={source}
                                            type="button"
                                            onClick={() => toggleSource(source)}
                                            className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                                                active
                                                    ? `${SOURCE_COLORS[source] || 'bg-blue-100 text-blue-700'} border-transparent`
                                                    : 'bg-slate-50 text-slate-400 border-slate-200 line-through'
                                            }`}
                                        >
                                            {source}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                )}
            </form>

            {/* ── Quick Search Chips ── */}
            {!searched && (
                <div className="mt-6">
                    <p className="text-xs font-medium text-slate-400 mb-2">QUICK SEARCH</p>
                    <div className="flex flex-wrap gap-2">
                        {QUICK_SEARCHES.map((item) => (
                            <button
                                key={item.label}
                                onClick={() => handleSearch(item.label)}
                                className="px-3.5 py-2 bg-white border border-slate-200 rounded-lg text-sm text-slate-600 hover:bg-blue-50 hover:text-blue-700 hover:border-blue-200 transition-all flex items-center gap-1.5"
                            >
                                <span>{item.icon}</span>
                                {item.label}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* ── Results Meta ── */}
            {searched && !loading && (
                <div className="mt-6 mb-4 flex items-center justify-between">
                    <p className="text-sm text-slate-500">
                        <span className="font-semibold text-slate-800">{meta.total_found || 0}</span> jobs found
                        {meta.keywords_used && meta.keywords_used.length > 0 && (
                            <span className="ml-2 text-xs text-slate-400">
                                Keywords: {meta.keywords_used.slice(0, 6).join(', ')}
                                {meta.keywords_used.length > 6 && '...'}
                            </span>
                        )}
                    </p>
                </div>
            )}

            {/* ── Loading Skeleton ── */}
            {loading && (
                <div className="mt-6 space-y-3">
                    {[1, 2, 3, 4].map((i) => (
                        <div key={i} className="bg-white p-5 rounded-xl border border-slate-200 animate-pulse">
                            <div className="h-5 bg-slate-200 rounded w-2/3 mb-3" />
                            <div className="flex gap-4">
                                <div className="h-4 bg-slate-100 rounded w-24" />
                                <div className="h-4 bg-slate-100 rounded w-20" />
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* ── Results ── */}
            {!loading && (
                <div className="mt-2 space-y-3">
                    {results.map((job, idx) => (
                        <div
                            key={idx}
                            className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm hover:shadow-md hover:border-slate-300 transition-all group"
                        >
                            <div className="flex justify-between items-start gap-4">
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 flex-wrap">
                                        <h3 className="font-semibold text-slate-900 truncate">{job.title}</h3>
                                        {job.source && (
                                            <span
                                                className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${
                                                    SOURCE_COLORS[job.source] || 'bg-slate-100 text-slate-600'
                                                }`}
                                            >
                                                {job.source}
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-4 mt-2 text-sm text-slate-500">
                                        {job.company && (
                                            <span className="flex items-center gap-1">
                                                <Briefcase size={14} /> {job.company}
                                            </span>
                                        )}
                                        <span className="flex items-center gap-1">
                                            <MapPin size={14} /> {job.location || 'N/A'}
                                        </span>
                                        {job.relevance != null && job.relevance > 0 && (
                                            <span
                                                className={`flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full ${relevanceColor(
                                                    job.relevance
                                                )}`}
                                            >
                                                <Star size={12} /> {job.relevance}%
                                            </span>
                                        )}
                                    </div>
                                </div>
                                <a
                                    href={job.link}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="shrink-0 bg-blue-50 text-blue-600 hover:bg-blue-600 hover:text-white px-4 py-2 rounded-lg flex items-center gap-1.5 text-sm font-medium transition-colors opacity-80 group-hover:opacity-100"
                                >
                                    Apply <ExternalLink size={14} />
                                </a>
                            </div>
                        </div>
                    ))}

                    {searched && !loading && results.length === 0 && (
                        <div className="text-center py-16">
                            <SearchIcon className="mx-auto text-slate-300 mb-3" size={40} />
                            <p className="text-slate-500 font-medium">No jobs found</p>
                            <p className="text-slate-400 text-sm mt-1">
                                Try broadening your search or adjusting the filters
                            </p>
                        </div>
                    )}

                    {!searched && (
                        <div className="text-center py-16">
                            <Globe className="mx-auto text-slate-300 mb-3" size={40} />
                            <p className="text-slate-500 font-medium">Search across multiple job boards</p>
                            <p className="text-slate-400 text-sm mt-1">
                                Enter keywords or click a quick search above to start
                            </p>
                        </div>
                    )}
                </div>
            )}
            </>
            )}

            {/* ══════════════════════════════════════════════════════════
                CV MATCH TAB
               ══════════════════════════════════════════════════════════ */}
            {tabMode === 'recommend' && (
            <>
                {/* Recommend Hero */}
                <div className="bg-gradient-to-br from-emerald-50 to-teal-50 border border-emerald-200 rounded-xl p-6 mb-6">
                    <div className="flex items-start gap-4">
                        <div className="bg-emerald-100 p-3 rounded-xl">
                            <FileText className="text-emerald-600" size={28} />
                        </div>
                        <div className="flex-1">
                            <h2 className="text-lg font-bold text-slate-800">CV-Based Job Matching</h2>
                            <p className="text-sm text-slate-600 mt-1">
                                Uses your uploaded CV to find the best matching jobs with an AI compatibility score.
                                Upload a CV on the <strong>Analyze</strong> page first, then click below.
                            </p>
                            <div className="flex items-center gap-3 mt-4 flex-wrap">
                                <button
                                    onClick={handleRecommend}
                                    disabled={recLoading}
                                    className="bg-emerald-600 text-white px-5 py-2.5 rounded-xl font-medium hover:bg-emerald-700 disabled:opacity-40 transition-colors flex items-center gap-2 text-sm"
                                >
                                    {recLoading ? (
                                        <Loader2 size={18} className="animate-spin" />
                                    ) : (
                                        <Zap size={18} />
                                    )}
                                    {recLoading ? 'Matching...' : 'Find Matching Jobs'}
                                </button>
                                {recMeta.cv_filename && (
                                    <span className="text-xs text-slate-500 flex items-center gap-1">
                                        <FileText size={14} /> CV: <strong>{recMeta.cv_filename}</strong>
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* CV Skills & Query Info */}
                {recMeta.cv_skills && recMeta.cv_skills.length > 0 && (
                    <div className="mb-4 flex flex-wrap items-center gap-2">
                        <span className="text-xs font-medium text-slate-400">DETECTED SKILLS:</span>
                        {recMeta.cv_skills.map((skill, i) => (
                            <span
                                key={i}
                                className="text-xs bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full border border-emerald-200"
                            >
                                {skill}
                            </span>
                        ))}
                    </div>
                )}

                {/* Error */}
                {recError && (
                    <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 mb-4 text-sm">
                        {recError}
                    </div>
                )}

                {/* Loading */}
                {recLoading && (
                    <div className="space-y-3">
                        {[1, 2, 3, 4].map((i) => (
                            <div key={i} className="bg-white p-5 rounded-xl border border-slate-200 animate-pulse">
                                <div className="flex gap-3 items-center mb-3">
                                    <div className="h-10 w-10 bg-emerald-100 rounded-lg" />
                                    <div className="flex-1">
                                        <div className="h-5 bg-slate-200 rounded w-2/3 mb-2" />
                                        <div className="h-4 bg-slate-100 rounded w-1/3" />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Meta */}
                {recSearched && !recLoading && (
                    <div className="mb-4 flex items-center justify-between">
                        <p className="text-sm text-slate-500">
                            <span className="font-semibold text-slate-800">{recMeta.total_found || 0}</span> matching jobs found
                        </p>
                    </div>
                )}

                {/* Results */}
                {!recLoading && (
                    <div className="space-y-3">
                        {recResults.map((job, idx) => (
                            <div
                                key={idx}
                                className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm hover:shadow-md hover:border-slate-300 transition-all group"
                            >
                                <div className="flex justify-between items-start gap-4">
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2 flex-wrap">
                                            <h3 className="font-semibold text-slate-900 truncate">{job.title}</h3>
                                            {job.source && (
                                                <span
                                                    className={`text-[11px] font-medium px-2 py-0.5 rounded-full ${
                                                        SOURCE_COLORS[job.source] || 'bg-slate-100 text-slate-600'
                                                    }`}
                                                >
                                                    {job.source}
                                                </span>
                                            )}
                                        </div>
                                        <div className="flex items-center gap-4 mt-2 text-sm text-slate-500">
                                            {job.company && (
                                                <span className="flex items-center gap-1">
                                                    <Briefcase size={14} /> {job.company}
                                                </span>
                                            )}
                                            <span className="flex items-center gap-1">
                                                <MapPin size={14} /> {job.location || 'N/A'}
                                            </span>
                                        </div>
                                        {job.description && (
                                            <p className="text-xs text-slate-400 mt-2 line-clamp-2">
                                                {job.description.slice(0, 150)}
                                                {job.description.length > 150 && '...'}
                                            </p>
                                        )}
                                    </div>
                                    <div className="flex flex-col items-end gap-2 shrink-0">
                                        {job.compatibility != null && (
                                            <span
                                                className={`flex items-center gap-1 text-sm font-bold px-3 py-1.5 rounded-lg border ${compatibilityColor(
                                                    job.compatibility
                                                )}`}
                                            >
                                                <TrendingUp size={14} /> {job.compatibility}%
                                            </span>
                                        )}
                                        <a
                                            href={job.link}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="bg-blue-50 text-blue-600 hover:bg-blue-600 hover:text-white px-4 py-2 rounded-lg flex items-center gap-1.5 text-sm font-medium transition-colors opacity-80 group-hover:opacity-100"
                                        >
                                            Apply <ExternalLink size={14} />
                                        </a>
                                    </div>
                                </div>
                            </div>
                        ))}

                        {recSearched && !recLoading && recResults.length === 0 && !recError && (
                            <div className="text-center py-16">
                                <SearchIcon className="mx-auto text-slate-300 mb-3" size={40} />
                                <p className="text-slate-500 font-medium">No matching jobs found</p>
                                <p className="text-slate-400 text-sm mt-1">
                                    Try uploading a different CV or broadening the source selection
                                </p>
                            </div>
                        )}

                        {!recSearched && (
                            <div className="text-center py-12">
                                <Zap className="mx-auto text-emerald-300 mb-3" size={40} />
                                <p className="text-slate-500 font-medium">Click "Find Matching Jobs" to start</p>
                                <p className="text-slate-400 text-sm mt-1">
                                    Make sure you've uploaded a CV on the Analyze page
                                </p>
                            </div>
                        )}
                    </div>
                )}
            </>
            )}
        </div>
    );
}
