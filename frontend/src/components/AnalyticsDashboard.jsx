import { useState, useEffect } from 'react';
import { api } from '../api';
import './AnalyticsDashboard.css';

export default function AnalyticsDashboard({ onClose }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadAnalytics = async () => {
            try {
                const result = await api.getAnalytics();
                setData(result);
            } catch (error) {
                console.error('Failed to load analytics:', error);
            } finally {
                setLoading(false);
            }
        };
        loadAnalytics();
    }, []);

    return (
        <div className="analytics-overlay">
            <div className="analytics-modal">
                <div className="analytics-header">
                    <h2>Model Performance Analytics</h2>
                    <button onClick={onClose} className="close-button">×</button>
                </div>

                <div className="analytics-content">
                    {loading ? (
                        <div className="analytics-loading">Loading stats...</div>
                    ) : (
                        <table className="analytics-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Avg Rank</th>
                                    <th>Win Rate</th>
                                    <th>Evaluations</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data?.models.length === 0 ? (
                                    <tr><td colSpan="4" className="no-data">No data available yet</td></tr>
                                ) : (
                                    data?.models.map(m => (
                                        <tr key={m.model}>
                                            <td className="model-name">{m.model}</td>
                                            <td>#{m.average_rank.toFixed(2)}</td>
                                            <td>{m.win_rate}%</td>
                                            <td>{m.evaluations}</td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>
        </div>
    );
}
