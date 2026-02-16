import { useState } from 'react';
import { Search as SearchIcon, MapPin, Briefcase, ExternalLink } from 'lucide-react';
import { searchJobs } from '../services/api';

interface Job {
    title: string;
    company: string;
    location: string;
    link: string;
    source?: string;
    relevance?: number;
}

export default function Search() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<Job[]>([]);
    const [loading, setLoading] = useState(false);

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return;

        setLoading(true);
        try {
            const data = await searchJobs(query);
            setResults(data.jobs || []);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-5xl mx-auto">
            <h1 className="text-2xl font-bold mb-6 text-slate-800">Job Search</h1>

            <form onSubmit={handleSearch} className="flex gap-3 mb-8">
                <div className="relative flex-1">
                    <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
                    <input
                        className="w-full pl-10 pr-4 py-3 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm"
                        placeholder="Search by role, keywords, or skills..."
                        value={query}
                        onChange={e => setQuery(e.target.value)}
                    />
                </div>
                <button
                    type="submit"
                    disabled={loading}
                    className="bg-blue-600 text-white px-6 py-3 rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
                >
                    {loading ? 'Searching...' : 'Find Jobs'}
                </button>
            </form>

            <div className="grid gap-4">
                {results.map((job, idx) => (
                    <div key={idx} className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex justify-between items-start">
                            <div>
                                <h3 className="font-semibold text-lg text-slate-900">{job.title}</h3>
                                <div className="flex items-center gap-4 mt-2 text-sm text-slate-500">
                                    <span className="flex items-center gap-1"><Briefcase size={16} /> {job.company}</span>
                                    <span className="flex items-center gap-1"><MapPin size={16} /> {job.location}</span>
                                </div>
                            </div>
                            <a
                                href={job.link}
                                target="_blank"
                                rel="noreferrer"
                                className="text-blue-600 hover:text-blue-800 flex items-center gap-1 text-sm font-medium"
                            >
                                Apply <ExternalLink size={14} />
                            </a>
                        </div>
                    </div>
                ))}

                {!loading && results.length === 0 && (
                    <div className="text-center py-12 text-slate-400">
                        Enter keywords to search for jobs across multiple remote boards.
                    </div>
                )}
            </div>
        </div>
    );
}
