import { useState } from 'react';
import { Upload, FileText, CheckCircle, Loader2 } from 'lucide-react';
import { uploadCV } from '../services/api';
import { cn } from '../lib/utils';

export default function Analyze() {
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError('');
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;

        setLoading(true);
        setError('');
        try {
            const data = await uploadCV(file);
            setResult(data);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to analyze CV');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <div>
                <h1 className="text-2xl font-bold text-slate-800">Analyze Candidates</h1>
                <p className="text-slate-500 mt-2">Upload a CV to extract skills and generate a summary.</p>
            </div>

            <div className="bg-white p-8 rounded-xl border border-dashed border-slate-300 text-center hover:bg-slate-50 transition-colors">
                <input
                    type="file"
                    id="cv-upload"
                    className="hidden"
                    accept=".pdf,.docx"
                    onChange={handleFileChange}
                />
                <label htmlFor="cv-upload" className="cursor-pointer flex flex-col items-center gap-4">
                    <div className="w-16 h-16 bg-blue-50 rounded-full flex items-center justify-center text-blue-600">
                        <Upload size={32} />
                    </div>
                    <div>
                        <span className="font-semibold text-blue-600">Click to upload</span>
                        <span className="text-slate-500"> or drag and drop</span>
                        <p className="text-sm text-slate-400 mt-1">PDF or DOCX (MAX. 5MB)</p>
                    </div>
                </label>
                {file && (
                    <div className="mt-4 flex items-center justify-center gap-2 text-sm text-slate-700 font-medium bg-slate-100 py-2 px-4 rounded-lg inline-block">
                        <FileText size={16} /> {file.name}
                    </div>
                )}
            </div>

            <button
                onClick={handleAnalyze}
                disabled={!file || loading}
                className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
            >
                {loading && <Loader2 className="animate-spin" size={20} />}
                {loading ? 'Analyzing...' : 'Start Analysis'}
            </button>

            {error && (
                <div className="p-4 bg-red-50 text-red-700 rounded-xl border border-red-200">
                    {error}
                </div>
            )}

            {result && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
                        <div className="flex items-center gap-3 mb-4">
                            <CheckCircle className="text-green-500" size={24} />
                            <h2 className="text-xl font-semibold text-slate-800">Analysis Complete</h2>
                        </div>

                        <div className="prose prose-slate max-w-none">
                            <h3 className="text-lg font-medium text-slate-900 mb-2">Professional Summary</h3>
                            <p className="text-slate-600 mb-6 bg-slate-50 p-4 rounded-lg border border-slate-100">
                                {result.summary}
                            </p>

                            <h3 className="text-lg font-medium text-slate-900 mb-3">Detected Skills</h3>
                            <div className="flex flex-wrap gap-2">
                                {result.skills_data?.skills?.map((skill: string, i: number) => (
                                    <span key={i} className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm font-medium border border-blue-100">
                                        {skill}
                                    </span>
                                ))}
                            </div>

                            <div className="grid grid-cols-2 gap-4 mt-6">
                                <div className="p-4 bg-slate-50 rounded-lg">
                                    <span className="text-sm text-slate-500 block mb-1">Experience</span>
                                    <span className="text-lg font-semibold text-slate-900">{result.skills_data?.experience_years || 0} years</span>
                                </div>
                                <div className="p-4 bg-slate-50 rounded-lg">
                                    <span className="text-sm text-slate-500 block mb-1">Education</span>
                                    <span className="text-lg font-semibold text-slate-900">{result.skills_data?.education?.length || 0} degrees</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
