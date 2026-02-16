import { MessageSquare, Search, FileText, Briefcase } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { cn } from '../lib/utils';

export default function Sidebar() {
    const location = useLocation();

    const links = [
        { name: 'Chat Assistant', path: '/', icon: MessageSquare },
        { name: 'Job Search', path: '/search', icon: Search },
        { name: 'Analyze CVs', path: '/analyze', icon: FileText },
        { name: 'Hiring Ops', path: '/hiring', icon: Briefcase },
    ];

    return (
        <div className="w-64 bg-white border-r border-slate-200 flex flex-col h-screen p-4">
            <div className="mb-8 px-2 flex items-center gap-2">
                <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">
                    IHR
                </div>
                <h1 className="font-bold text-xl text-slate-800">Intelligent HR</h1>
            </div>

            <nav className="flex-1 space-y-1">
                {links.map((link) => {
                    const Icon = link.icon;
                    const isActive = location.pathname === link.path;

                    return (
                        <Link
                            key={link.path}
                            to={link.path}
                            className={cn(
                                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                                isActive
                                    ? "bg-blue-50 text-blue-700"
                                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                            )}
                        >
                            <Icon size={18} />
                            {link.name}
                        </Link>
                    );
                })}
            </nav>

            <div className="mt-auto px-2 pt-4 border-t border-slate-100">
                <p className="text-xs text-slate-400">v2.0 Professional</p>
            </div>
        </div>
    );
}
