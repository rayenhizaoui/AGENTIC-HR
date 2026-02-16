import { useState, useCallback } from 'react';
import { generateOffer, checkSalary } from '../services/api';
import {
    Loader2,
    Copy,
    Check,
    Briefcase,
    User,
    DollarSign,
    Calendar,
    MapPin,
    FileText,
    TrendingUp,
    TrendingDown,
    CheckCircle2,
    AlertTriangle,
    Download,
    Sparkles,
    Building2,
    ChevronRight,
    RotateCcw,
} from 'lucide-react';

/* ────────────────────────── Types ────────────────────────── */

interface SalaryResult {
    within_range: boolean | null;
    market_min: number | null;
    market_max: number | null;
    market_median: number | null;
    deviation_percent: number | null;
    flag: string;
    recommendation: string;
}

interface FormData {
    candidate_name: string;
    role: string;
    department: string;
    location: string;
    contract_type: string;
    salary: string;
    currency: string;
    start_date: string;
    hiring_manager: string;
}

type Step = 'details' | 'compensation' | 'review';

const STEPS: { key: Step; label: string; icon: React.ElementType }[] = [
    { key: 'details', label: 'Position Details', icon: Briefcase },
    { key: 'compensation', label: 'Compensation', icon: DollarSign },
    { key: 'review', label: 'Review & Generate', icon: FileText },
];

const CONTRACT_TYPES = ['Full-time', 'Part-time', 'Contract', 'Internship', 'Freelance'];
const DEPARTMENTS = ['Engineering', 'Data Science', 'Product', 'Design', 'Marketing', 'Operations', 'HR', 'Finance'];
const CURRENCIES = ['USD', 'EUR', 'TND', 'GBP', 'CAD'];

/* ────────── Reusable sub-components (outside main component) ────────── */

function FormInput({ label, icon: Icon, ...props }: any) {
    return (
        <div>
            <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">{label}</label>
            <div className="relative">
                {Icon && <Icon className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />}
                <input
                    {...props}
                    className={`w-full ${Icon ? 'pl-10' : 'pl-3'} pr-3 py-2.5 border border-slate-200 rounded-xl text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-shadow`}
                />
            </div>
        </div>
    );
}

function FormSelect({ label, icon: Icon, options, ...props }: any) {
    return (
        <div>
            <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">{label}</label>
            <div className="relative">
                {Icon && <Icon className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />}
                <select
                    {...props}
                    className={`w-full ${Icon ? 'pl-10' : 'pl-3'} pr-3 py-2.5 border border-slate-200 rounded-xl text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none`}
                >
                    {options.map((o: string) => <option key={o} value={o}>{o}</option>)}
                </select>
            </div>
        </div>
    );
}

function StepperBar({ step, currentIdx, onStepClick }: { step: Step; currentIdx: number; onStepClick: (s: Step) => void }) {
    return (
        <div className="flex items-center gap-1 mb-8">
            {STEPS.map((s, i) => {
                const Icon = s.icon;
                const isActive = s.key === step;
                const isDone = i < currentIdx;
                return (
                    <div key={s.key} className="flex items-center gap-1 flex-1">
                        <button
                            onClick={() => { if (isDone) onStepClick(s.key); }}
                            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all w-full ${
                                isActive
                                    ? 'bg-blue-600 text-white shadow-md'
                                    : isDone
                                        ? 'bg-emerald-50 text-emerald-700 hover:bg-emerald-100 cursor-pointer'
                                        : 'bg-slate-100 text-slate-400 cursor-default'
                            }`}
                        >
                            {isDone ? <CheckCircle2 size={16} /> : <Icon size={16} />}
                            <span className="hidden sm:inline">{s.label}</span>
                        </button>
                        {i < STEPS.length - 1 && <ChevronRight size={16} className="text-slate-300 shrink-0" />}
                    </div>
                );
            })}
        </div>
    );
}

function SalaryGauge({ result, salaryStr }: { result: SalaryResult; salaryStr: string }) {
    if (!result.market_min || !result.market_max || !result.market_median) {
        return (
            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-700 flex items-start gap-2">
                <AlertTriangle size={18} className="shrink-0 mt-0.5" />
                <span>{result.recommendation}</span>
            </div>
        );
    }

    const salaryNum = parseFloat(salaryStr.replace(/[^0-9.]/g, ''));
    const range = result.market_max - result.market_min;
    const pct = Math.min(Math.max(((salaryNum - result.market_min) / range) * 100, 0), 100);

    const flagColors: Record<string, string> = {
        ok: 'bg-emerald-50 border-emerald-200 text-emerald-700',
        low: 'bg-red-50 border-red-200 text-red-700',
        high: 'bg-amber-50 border-amber-200 text-amber-700',
    };

    return (
        <div className={`border rounded-xl p-4 space-y-3 ${flagColors[result.flag] || flagColors.ok}`}>
            <div className="flex items-center justify-between text-sm font-medium">
                <span className="flex items-center gap-1.5">
                    {result.flag === 'ok' && <CheckCircle2 size={16} />}
                    {result.flag === 'low' && <TrendingDown size={16} />}
                    {result.flag === 'high' && <TrendingUp size={16} />}
                    Market Salary Check
                </span>
                <span className="text-xs font-normal opacity-70">
                    {result.deviation_percent != null && `${result.deviation_percent > 0 ? '+' : ''}${result.deviation_percent}% from median`}
                </span>
            </div>

            {/* Gauge bar */}
            <div className="relative h-3 bg-white/50 rounded-full overflow-hidden border border-current/10">
                <div className="absolute inset-0 flex">
                    <div className="bg-red-200/60 h-full" style={{ width: '20%' }} />
                    <div className="bg-emerald-200/60 h-full" style={{ width: '60%' }} />
                    <div className="bg-amber-200/60 h-full" style={{ width: '20%' }} />
                </div>
                <div
                    className="absolute top-0 w-3 h-3 bg-slate-800 rounded-full border-2 border-white shadow-md -translate-x-1/2"
                    style={{ left: `${pct}%` }}
                />
            </div>

            <div className="flex justify-between text-[11px] font-mono opacity-60">
                <span>${result.market_min.toLocaleString()}</span>
                <span>Median: ${result.market_median.toLocaleString()}</span>
                <span>${result.market_max.toLocaleString()}</span>
            </div>

            <p className="text-xs leading-relaxed">{result.recommendation}</p>
        </div>
    );
}

function ReviewCard({ form }: { form: FormData }) {
    return (
        <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                <FileText size={16} className="text-blue-500" /> Offer Summary
            </h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
                {[
                    { label: 'Candidate', value: form.candidate_name, icon: User },
                    { label: 'Position', value: form.role, icon: Briefcase },
                    { label: 'Department', value: form.department, icon: Building2 },
                    { label: 'Location', value: form.location, icon: MapPin },
                    { label: 'Salary', value: `${form.salary} ${form.currency}`, icon: DollarSign },
                    { label: 'Start Date', value: form.start_date || 'TBD', icon: Calendar },
                    { label: 'Contract', value: form.contract_type, icon: FileText },
                    { label: 'Hiring Manager', value: form.hiring_manager || 'TBD', icon: User },
                ].map(item => (
                    <div key={item.label} className="flex items-start gap-2 bg-slate-50 rounded-lg px-3 py-2">
                        <item.icon size={14} className="text-slate-400 mt-0.5 shrink-0" />
                        <div>
                            <p className="text-[11px] text-slate-400 uppercase tracking-wide">{item.label}</p>
                            <p className="font-medium text-slate-700 truncate">{item.value || '—'}</p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

/* ──────────────────────── Main Component ─────────────────────── */

export default function Hiring() {
    const [step, setStep] = useState<Step>('details');

    const [form, setForm] = useState<FormData>({
        candidate_name: '',
        role: '',
        department: 'Engineering',
        location: 'Tunis, Tunisia',
        contract_type: 'Full-time',
        salary: '',
        currency: 'USD',
        start_date: '',
        hiring_manager: '',
    });

    const [offer, setOffer] = useState('');
    const [loading, setLoading] = useState(false);
    const [copied, setCopied] = useState(false);
    const [error, setError] = useState('');

    // Salary check
    const [salaryResult, setSalaryResult] = useState<SalaryResult | null>(null);
    const [salaryLoading, setSalaryLoading] = useState(false);

    /* ── Field updater (stable via useCallback + functional setState) ── */
    const set = useCallback(
        (field: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
            setForm(prev => ({ ...prev, [field]: e.target.value })),
        []
    );

    /* ── Salary check ── */
    const handleSalaryCheck = async () => {
        if (!form.role || !form.salary) return;
        setSalaryLoading(true);
        try {
            const numSalary = parseFloat(form.salary.replace(/[^0-9.]/g, ''));
            if (isNaN(numSalary)) return;
            const result = await checkSalary(form.role, numSalary);
            setSalaryResult(result);
        } catch {
            setSalaryResult(null);
        } finally {
            setSalaryLoading(false);
        }
    };

    /* ── Generate offer ── */
    const handleGenerate = async () => {
        setLoading(true);
        setError('');
        try {
            const data = await generateOffer(form);
            setOffer(data.offer_letter);
        } catch (err: any) {
            setError(err?.response?.data?.detail || 'Failed to generate offer. Is the LLM running?');
        } finally {
            setLoading(false);
        }
    };

    /* ── Copy ── */
    const copyToClipboard = () => {
        navigator.clipboard.writeText(offer);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    /* ── Download as .txt ── */
    const downloadOffer = () => {
        const blob = new Blob([offer], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Offer_${form.candidate_name.replace(/\s+/g, '_') || 'Letter'}.txt`;
        a.click();
        URL.revokeObjectURL(url);
    };

    /* ── Reset ── */
    const resetAll = () => {
        setOffer('');
        setError('');
        setSalaryResult(null);
        setStep('details');
    };

    /* ── Step navigation ── */
    const currentIdx = STEPS.findIndex(s => s.key === step);
    const canNext = () => {
        if (step === 'details') return form.candidate_name.trim() && form.role.trim();
        if (step === 'compensation') return form.salary.trim();
        return true;
    };

    /* ────────────────── RENDER ────────────────── */
    return (
        <div className="max-w-5xl mx-auto">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
                    <Briefcase className="text-blue-600" size={28} />
                    Hiring Operations
                </h1>
                <p className="text-slate-500 text-sm mt-1">
                    Generate professional offer letters with AI-powered salary insights
                </p>
            </div>

            {/* If offer generated → show result */}
            {offer ? (
                <div className="space-y-4">
                    {/* Actions bar */}
                    <div className="flex items-center justify-between bg-emerald-50 border border-emerald-200 rounded-xl px-5 py-3">
                        <span className="text-sm font-medium text-emerald-700 flex items-center gap-2">
                            <CheckCircle2 size={18} /> Offer letter generated successfully
                        </span>
                        <div className="flex items-center gap-2">
                            <button
                                onClick={copyToClipboard}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
                            >
                                {copied ? <Check size={14} className="text-emerald-600" /> : <Copy size={14} />}
                                {copied ? 'Copied' : 'Copy'}
                            </button>
                            <button
                                onClick={downloadOffer}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
                            >
                                <Download size={14} /> Download
                            </button>
                            <button
                                onClick={resetAll}
                                className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                            >
                                <RotateCcw size={14} /> New Offer
                            </button>
                        </div>
                    </div>

                    {/* Offer letter preview */}
                    <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
                        <div className="bg-slate-800 px-6 py-3 flex items-center gap-2">
                            <FileText size={16} className="text-slate-400" />
                            <span className="text-sm font-medium text-slate-200">
                                Offer Letter — {form.candidate_name} — {form.role}
                            </span>
                        </div>
                        <div className="p-8 max-h-[600px] overflow-y-auto">
                            <div className="whitespace-pre-wrap font-serif text-slate-800 leading-relaxed text-[15px]">
                                {offer}
                            </div>
                        </div>
                    </div>
                </div>
            ) : (
                /* ── Stepper form ── */
                <>
                    <StepperBar step={step} currentIdx={currentIdx} onStepClick={setStep} />

                    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                        {/* Left: Form (3 cols) */}
                        <div className="lg:col-span-3">
                            <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 space-y-5">

                                {/* STEP 1: Position Details */}
                                {step === 'details' && (
                                    <>
                                        <div className="flex items-center gap-2 mb-1">
                                            <Briefcase size={18} className="text-blue-600" />
                                            <h2 className="text-lg font-semibold text-slate-800">Position Details</h2>
                                        </div>

                                        <FormInput label="Candidate Full Name" icon={User} placeholder="e.g. Rayen Hizaoui"
                                            value={form.candidate_name} onChange={set('candidate_name')} required />

                                        <FormInput label="Role / Position" icon={Briefcase} placeholder="e.g. Senior AI Engineer"
                                            value={form.role} onChange={set('role')} required />

                                        <div className="grid grid-cols-2 gap-4">
                                            <FormSelect label="Department" icon={Building2}
                                                options={DEPARTMENTS} value={form.department} onChange={set('department')} />
                                            <FormInput label="Location" icon={MapPin} placeholder="e.g. Tunis, Tunisia"
                                                value={form.location} onChange={set('location')} />
                                        </div>

                                        <FormSelect label="Contract Type" icon={FileText}
                                            options={CONTRACT_TYPES} value={form.contract_type} onChange={set('contract_type')} />
                                    </>
                                )}

                                {/* STEP 2: Compensation */}
                                {step === 'compensation' && (
                                    <>
                                        <div className="flex items-center gap-2 mb-1">
                                            <DollarSign size={18} className="text-blue-600" />
                                            <h2 className="text-lg font-semibold text-slate-800">Compensation</h2>
                                        </div>

                                        <div className="grid grid-cols-3 gap-4">
                                            <div className="col-span-2">
                                                <FormInput label="Annual Salary" icon={DollarSign} placeholder="e.g. 85000"
                                                    value={form.salary} onChange={set('salary')} required />
                                            </div>
                                            <FormSelect label="Currency" options={CURRENCIES}
                                                value={form.currency} onChange={set('currency')} />
                                        </div>

                                        {/* Salary check button */}
                                        <button
                                            type="button"
                                            onClick={handleSalaryCheck}
                                            disabled={salaryLoading || !form.role || !form.salary}
                                            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-medium bg-slate-100 text-slate-700 rounded-xl hover:bg-slate-200 disabled:opacity-40 transition-colors border border-slate-200"
                                        >
                                            {salaryLoading ? <Loader2 size={16} className="animate-spin" /> : <TrendingUp size={16} />}
                                            Check Market Salary Range
                                        </button>

                                        {salaryResult && <SalaryGauge result={salaryResult} salaryStr={form.salary} />}

                                        <div className="grid grid-cols-2 gap-4">
                                            <FormInput label="Start Date" icon={Calendar} type="date"
                                                value={form.start_date} onChange={set('start_date')} />
                                            <FormInput label="Hiring Manager" icon={User} placeholder="e.g. John Doe"
                                                value={form.hiring_manager} onChange={set('hiring_manager')} />
                                        </div>
                                    </>
                                )}

                                {/* STEP 3: Review & Generate */}
                                {step === 'review' && (
                                    <>
                                        <div className="flex items-center gap-2 mb-1">
                                            <Sparkles size={18} className="text-blue-600" />
                                            <h2 className="text-lg font-semibold text-slate-800">Review & Generate</h2>
                                        </div>

                                        <ReviewCard form={form} />

                                        {error && (
                                            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700 flex items-start gap-2">
                                                <AlertTriangle size={16} className="shrink-0 mt-0.5" />
                                                {error}
                                            </div>
                                        )}

                                        <button
                                            onClick={handleGenerate}
                                            disabled={loading}
                                            className="w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-3 rounded-xl font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors shadow-md"
                                        >
                                            {loading ? (
                                                <>
                                                    <Loader2 size={18} className="animate-spin" />
                                                    Generating with AI...
                                                </>
                                            ) : (
                                                <>
                                                    <Sparkles size={18} />
                                                    Generate Offer Letter
                                                </>
                                            )}
                                        </button>
                                    </>
                                )}

                                {/* Navigation buttons */}
                                <div className="flex justify-between pt-2 border-t border-slate-100">
                                    <button
                                        type="button"
                                        onClick={() => setStep(STEPS[currentIdx - 1]?.key || 'details')}
                                        disabled={currentIdx === 0}
                                        className="px-4 py-2 text-sm font-medium text-slate-500 hover:text-slate-700 disabled:opacity-30 transition-colors"
                                    >
                                        Back
                                    </button>
                                    {step !== 'review' && (
                                        <button
                                            type="button"
                                            onClick={() => setStep(STEPS[currentIdx + 1]?.key || 'review')}
                                            disabled={!canNext()}
                                            className="px-5 py-2 text-sm font-medium bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-40 transition-colors flex items-center gap-1"
                                        >
                                            Continue <ChevronRight size={16} />
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Right: Live summary (2 cols) */}
                        <div className="lg:col-span-2 space-y-4">
                            {/* Live preview card */}
                            <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 sticky top-6">
                                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                                    Live Preview
                                </h3>
                                <div className="space-y-3 text-sm">
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 font-bold text-lg">
                                            {form.candidate_name ? form.candidate_name[0].toUpperCase() : '?'}
                                        </div>
                                        <div>
                                            <p className="font-semibold text-slate-800">{form.candidate_name || 'Candidate Name'}</p>
                                            <p className="text-slate-400 text-xs">{form.role || 'Position'}</p>
                                        </div>
                                    </div>

                                    <div className="border-t border-slate-100 pt-3 space-y-2">
                                        {[
                                            { icon: Building2, label: 'Dept', value: form.department },
                                            { icon: MapPin, label: 'Location', value: form.location },
                                            { icon: FileText, label: 'Contract', value: form.contract_type },
                                            { icon: DollarSign, label: 'Salary', value: form.salary ? `${form.salary} ${form.currency}` : '—' },
                                            { icon: Calendar, label: 'Start', value: form.start_date || '—' },
                                        ].map(item => (
                                            <div key={item.label} className="flex items-center justify-between text-xs">
                                                <span className="flex items-center gap-1.5 text-slate-400">
                                                    <item.icon size={13} /> {item.label}
                                                </span>
                                                <span className="text-slate-700 font-medium">{item.value || '—'}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Completion indicator */}
                                <div className="mt-4 pt-3 border-t border-slate-100">
                                    <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
                                        <span>Completion</span>
                                        <span>{Math.round(
                                            ([form.candidate_name, form.role, form.department, form.location, form.salary, form.start_date, form.hiring_manager]
                                                .filter(Boolean).length / 7) * 100
                                        )}%</span>
                                    </div>
                                    <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-blue-500 rounded-full transition-all duration-300"
                                            style={{
                                                width: `${([form.candidate_name, form.role, form.department, form.location, form.salary, form.start_date, form.hiring_manager]
                                                    .filter(Boolean).length / 7) * 100}%`
                                            }}
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
