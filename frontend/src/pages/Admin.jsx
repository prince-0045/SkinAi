import React, { useState, useEffect } from 'react';
import { Users, Activity, Lock, RefreshCw, ChevronRight, User as UserIcon } from 'lucide-react';

const Admin = () => {
  const [password, setPassword] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const API_BASE_URL = import.meta.env.VITE_API_URL || '';

  const fetchStats = async (pw) => {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/admin/stats`, {
        headers: {
          'X-Admin-Password': pw || password
        }
      });

      if (!response.ok) {
        throw new Error('Invalid Admin Password or Server Error');
      }

      const data = await response.json();
      setStats(data);
      setIsAuthenticated(true);
      localStorage.setItem('admin_session_pw', pw || password);
    } catch (err) {
      setError(err.message);
      setIsAuthenticated(false);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const savedPw = localStorage.getItem('admin_session_pw');
    if (savedPw) {
      fetchStats(savedPw);
    }
  }, []);

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen pt-24 pb-12 px-4 flex items-center justify-center bg-gray-50">
        <div className="bg-white p-8 rounded-2xl shadow-xl w-full max-w-md border border-gray-100">
          <div className="flex justify-center mb-6">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
              <Lock className="text-blue-600 w-8 h-8" />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-center mb-2 text-gray-800">Admin Access</h1>
          <p className="text-gray-500 text-center mb-8">Please enter your admin password to continue</p>
          
          <form onSubmit={(e) => { e.preventDefault(); fetchStats(); }}>
            <input
              type="password"
              className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all mb-4 outline-none"
              placeholder="Admin Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            {error && <p className="text-red-500 text-sm mb-4 text-center">{error}</p>}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-[#378ADD] text-white py-3 rounded-xl font-semibold hover:bg-blue-600 transition-colors shadow-lg flex items-center justify-center gap-2"
            >
              {loading ? <RefreshCw className="animate-spin" /> : 'Enter Dashboard'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-24 pb-12 px-4 md:px-8 max-w-7xl mx-auto">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-gray-900 font-syne">DermAura Admin</h1>
          <p className="text-gray-500">Global Overview & Statistics</p>
        </div>
        <button 
          onClick={() => fetchStats()}
          className="flex items-center gap-2 bg-white border border-gray-200 px-4 py-2 rounded-xl text-gray-600 hover:bg-gray-50 transition-all font-medium"
        >
          <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
          Refresh Stats
        </button>
      </div>

      {stats && (
        <>
          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
            <div className="bg-white p-6 rounded-2xl shadow-md border border-gray-100 flex items-center gap-5">
              <div className="w-14 h-14 bg-blue-50 rounded-2xl flex items-center justify-center text-blue-600">
                <Users size={28} />
              </div>
              <div>
                <p className="text-gray-500 font-medium text-sm">Total Registered Users</p>
                <h2 className="text-3xl font-bold text-gray-900">{stats.total_users}</h2>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-md border border-gray-100 flex items-center gap-5">
              <div className="w-14 h-14 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600">
                <Activity size={28} />
              </div>
              <div>
                <p className="text-gray-500 font-medium text-sm">Active (Last 24h)</p>
                <h2 className="text-3xl font-bold text-gray-900">{stats.active_users_24h}</h2>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-md border border-gray-100 flex items-center gap-5 relative overflow-hidden">
              <div className="w-14 h-14 bg-emerald-50 rounded-2xl flex items-center justify-center text-emerald-600">
                <div className="relative">
                  <Activity size={28} />
                  <span className="absolute -top-1 -right-1 flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                  </span>
                </div>
              </div>
              <div>
                <p className="text-gray-500 font-medium text-sm">Live Now</p>
                <h2 className="text-3xl font-bold text-gray-900">{stats.live_users}</h2>
              </div>
              <div className="absolute top-0 right-0 p-2">
                 <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest bg-emerald-50 px-2 py-0.5 rounded-full">Real-time</span>
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-2xl shadow-md border border-gray-100 flex items-center gap-5">
              <div className="w-14 h-14 bg-green-50 rounded-2xl flex items-center justify-center text-green-600">
                <Activity size={28} />
              </div>
              <div>
                <p className="text-gray-500 font-medium text-sm">Total Skin Scans</p>
                <h2 className="text-3xl font-bold text-gray-900">{stats.total_scans}</h2>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Recent Users */}
            <div className="bg-white rounded-2xl shadow-md border border-gray-100 overflow-hidden">
              <div className="p-6 border-b border-gray-50 flex justify-between items-center bg-gray-50/50">
                <h3 className="font-bold text-gray-800 flex items-center gap-2">
                  <UserIcon size={20} className="text-blue-500" />
                  Recent Signups
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="bg-gray-50 text-gray-500 text-xs uppercase tracking-wider">
                      <th className="px-6 py-4 font-semibold">User</th>
                      <th className="px-6 py-4 font-semibold">Joined Date</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {stats.recent_users.map((u, i) => (
                      <tr key={i} className="hover:bg-blue-50/30 transition-colors">
                        <td className="px-6 py-4">
                          <div className="font-medium text-gray-800">{u.name}</div>
                          <div className="text-sm text-gray-500">{u.email}</div>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-500">
                          {new Date(u.created_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Recent Scans */}
            <div className="bg-white rounded-2xl shadow-md border border-gray-100 overflow-hidden">
              <div className="p-6 border-b border-gray-50 flex justify-between items-center bg-gray-50/50">
                <h3 className="font-bold text-gray-800 flex items-center gap-2">
                  <Activity size={20} className="text-green-500" />
                  Recent Scans
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="bg-gray-50 text-gray-500 text-xs uppercase tracking-wider">
                      <th className="px-6 py-4 font-semibold">Detected Disease</th>
                      <th className="px-6 py-4 font-semibold">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {stats.recent_scans.map((s, i) => (
                      <tr key={i} className="hover:bg-green-50/30 transition-colors">
                        <td className="px-6 py-4">
                          <div className="font-medium text-gray-800">{s.disease}</div>
                          <div className="text-xs text-gray-400">{new Date(s.created_at).toLocaleString()}</div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <div className="w-full bg-gray-100 h-2 rounded-full overflow-hidden max-w-[60px]">
                              <div 
                                className="h-full bg-blue-500" 
                                style={{ width: `${s.confidence * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-semibold text-gray-700">
                              {(s.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Admin;
