import { useState } from 'react';
import { generateOffer } from '../services/api';
import { Loader2, Copy, Check } from 'lucide-react';

export default function Hiring() {
    const [formData, setFormData] = useState({
        candidate_name: '',
        role: '',
        department: '',
        salary: '',
        start_date: ''
    });
    const [offer, setOffer] = useState('');
    const [loading, setLoading] = useState(false);
    const [copied, setCopied] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        try {
            const data = await generateOffer(formData);
            setOffer(data.offer_letter);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const copyToClipboard = () => {
        navigator.clipboard.writeText(offer);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="max-w-4xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
                <h1 className="text-2xl font-bold text-slate-800 mb-6">Draft Offer Letter</h1>

                <form onSubmit={handleSubmit} className="space-y-4 bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Candidate Name</label>
                        <input
                            required
                            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            value={formData.candidate_name}
                            onChange={e => setFormData({ ...formData, candidate_name: e.target.value })}
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Role / Position</label>
                        <input
                            required
                            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            value={formData.role}
                            onChange={e => setFormData({ ...formData, role: e.target.value })}
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Department</label>
                            <input
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                value={formData.department}
                                onChange={e => setFormData({ ...formData, department: e.target.value })}
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-slate-700 mb-1">Salary</label>
                            <input
                                className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                                value={formData.salary}
                                onChange={e => setFormData({ ...formData, salary: e.target.value })}
                            />
                        </div>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-1">Start Date</label>
                        <input
                            type="date"
                            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            value={formData.start_date}
                            onChange={e => setFormData({ ...formData, start_date: e.target.value })}
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-slate-900 text-white py-2.5 rounded-lg font-medium hover:bg-slate-800 disabled:opacity-50 transition-colors flex justify-center items-center gap-2 mt-2"
                    >
                        {loading ? <Loader2 className="animate-spin" size={18} /> : 'Generate Draft'}
                    </button>
                </form>
            </div>

            <div className="flex flex-col h-full">
                <h2 className="text-xl font-bold text-slate-800 mb-6">Preview</h2>
                <div className="flex-1 bg-white border border-slate-200 rounded-xl p-6 shadow-sm relative min-h-[400px]">
                    {offer ? (
                        <>
                            <button
                                onClick={copyToClipboard}
                                className="absolute top-4 right-4 p-2 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-md transition-colors"
                                title="Copy to clipboard"
                            >
                                {copied ? <Check size={18} /> : <Copy size={18} />}
                            </button>
                            <div className="whitespace-pre-wrap font-serif text-slate-800 leading-relaxed">
                                {offer}
                            </div>
                        </>
                    ) : (
                        <div className="h-full flex items-center justify-center text-slate-400 text-sm">
                            Fill out the form to generate an offer letter.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
