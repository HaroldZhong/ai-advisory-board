import { useState, useEffect } from 'react';
import { api } from '../api';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import { Loader2 } from "lucide-react";

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
        <Dialog open={true} onOpenChange={onClose}>
            <DialogContent className="max-w-3xl">
                <DialogHeader>
                    <DialogTitle>Model Performance Analytics</DialogTitle>
                </DialogHeader>

                <div className="py-4">
                    {loading ? (
                        <div className="flex justify-center items-center h-40 text-muted-foreground">
                            <Loader2 className="h-6 w-6 animate-spin mr-2" />
                            Loading stats...
                        </div>
                    ) : (
                        <div className="rounded-md border">
                            <table className="w-full caption-bottom text-sm">
                                <thead className="[&_tr]:border-b">
                                    <tr className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                                        <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Model</th>
                                        <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Avg Rank</th>
                                        <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Win Rate</th>
                                        <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Evaluations</th>
                                    </tr>
                                </thead>
                                <tbody className="[&_tr:last-child]:border-0">
                                    {data?.models.length === 0 ? (
                                        <tr><td colSpan="4" className="p-4 text-center text-muted-foreground">No data available yet</td></tr>
                                    ) : (
                                        data?.models.map(m => (
                                            <tr key={m.model} className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                                                <td className="p-4 align-middle font-medium">{m.model}</td>
                                                <td className="p-4 align-middle">#{m.average_rank.toFixed(2)}</td>
                                                <td className="p-4 align-middle">{m.win_rate}%</td>
                                                <td className="p-4 align-middle">{m.evaluations}</td>
                                            </tr>
                                        ))
                                    )}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    );
}
