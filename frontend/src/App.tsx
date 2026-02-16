import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Chat from './pages/Chat';
import Search from './pages/Search';
import Analyze from './pages/Analyze';
import Hiring from './pages/Hiring';

function App() {
  return (
    <Router>
      <div className="flex min-h-screen bg-slate-50 text-slate-900">
        <Sidebar />
        <main className="flex-1 p-8 overflow-y-auto h-screen">
          <Routes>
            <Route path="/" element={<Chat />} />
            <Route path="/search" element={<Search />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/hiring" element={<Hiring />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
