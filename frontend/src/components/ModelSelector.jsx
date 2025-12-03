import React, { useState, useEffect } from 'react';
import { api } from '../api';
import './ModelSelector.css';

const DEFAULT_COUNCIL = [];

export default function ModelSelector({
    isOpen,
    onClose,
    onConfirm,
    initialCouncil = DEFAULT_COUNCIL,
    initialChairman = ''
}) {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedCouncil, setSelectedCouncil] = useState([]);
    const [selectedChairman, setSelectedChairman] = useState('');
    const [error, setError] = useState(null);

    useEffect(() => {
        if (isOpen) {
            fetchModels();
            setSelectedCouncil(initialCouncil.length > 0 ? initialCouncil : []);
            setSelectedChairman(initialChairman);
        }
    }, [isOpen, initialCouncil, initialChairman]);

    const fetchModels = async () => {
        try {
            setLoading(true);
            const data = await api.getModels();
            setModels(data.models);

            // Set defaults if nothing selected
            if (initialCouncil.length === 0 && data.models.length > 0) {
                // Default to first 5 council-capable models
                const defaults = data.models
                    .filter(m => m.type === 'council' || m.type === 'both')
                    .slice(0, 5)
                    .map(m => m.id);
                setSelectedCouncil(defaults);
            }

            if (!initialChairman && data.models.length > 0) {
                // Default to first chairman-capable model
                const defaultChair = data.models.find(m => m.type === 'chairman' || m.type === 'both');
                if (defaultChair) setSelectedChairman(defaultChair.id);
            }

        } catch (err) {
            setError('Failed to load models');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleCouncilToggle = (modelId) => {
        setSelectedCouncil(prev => {
            if (prev.includes(modelId)) {
                return prev.filter(id => id !== modelId);
            } else {
                if (prev.length >= 8) return prev; // Max 8
                return [...prev, modelId];
            }
        });
    };

    const handleConfirm = () => {
        if (selectedCouncil.length < 5) {
            alert('Please select at least 5 council members.');
            return;
        }
        if (!selectedChairman) {
            alert('Please select a Chairman.');
            return;
        }
        onConfirm(selectedCouncil, selectedChairman);
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="modal-overlay">
            <div className="modal-content model-selector">
                <div className="modal-header">
                    <h2>Select Council & Chairman</h2>
                    <button className="close-button" onClick={onClose}>×</button>
                </div>

                {loading ? (
                    <div className="loading-spinner">Loading models...</div>
                ) : error ? (
                    <div className="error-message">{error}</div>
                ) : (
                    <div className="selector-body">
                        <div className="section">
                            <h3>
                                Chairman
                                <span className="subtitle">(Select 1)</span>
                            </h3>
                            <div className="model-grid">
                                {models.filter(m => m.type === 'chairman' || m.type === 'both').map(model => (
                                    <label key={`chair-${model.id}`} className={`model-card ${selectedChairman === model.id ? 'selected' : ''}`}>
                                        <input
                                            type="radio"
                                            name="chairman"
                                            checked={selectedChairman === model.id}
                                            onChange={() => setSelectedChairman(model.id)}
                                        />
                                        <div className="model-info">
                                            <div className="model-name">{model.name}</div>
                                            <div className="model-meta">
                                                <span className="price">
                                                    In: ${model.pricing.input}/M | Out: ${model.pricing.output}/M
                                                </span>
                                                <div className="capabilities">
                                                    {model.capabilities.map(c => (
                                                        <span key={c} className="tag">{c}</span>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    </label>
                                ))}
                            </div>
                        </div>

                        <div className="section">
                            <h3>
                                Council Members
                                <span className="subtitle">(Select 5-8)</span>
                                <span className="count-badge">{selectedCouncil.length}/8</span>
                            </h3>
                            <div className="model-grid">
                                {models.filter(m => m.type === 'council' || m.type === 'both').map(model => (
                                    <label key={`council-${model.id}`} className={`model-card ${selectedCouncil.includes(model.id) ? 'selected' : ''}`}>
                                        <input
                                            type="checkbox"
                                            checked={selectedCouncil.includes(model.id)}
                                            onChange={() => handleCouncilToggle(model.id)}
                                            disabled={!selectedCouncil.includes(model.id) && selectedCouncil.length >= 8}
                                        />
                                        <div className="model-info">
                                            <div className="model-name">{model.name}</div>
                                            <div className="model-meta">
                                                <span className="price">
                                                    In: ${model.pricing.input}/M | Out: ${model.pricing.output}/M
                                                </span>
                                                <div className="capabilities">
                                                    {model.capabilities.map(c => (
                                                        <span key={c} className="tag">{c}</span>
                                                    ))}
                                                </div>
                                            </div>
                                        </div>
                                    </label>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                <div className="modal-footer">
                    <div className="budget-estimate">
                        {/* Placeholder for budget logic if needed here, or just keep it simple */}
                    </div>
                    <div className="actions">
                        <button className="cancel-button" onClick={onClose}>Cancel</button>
                        <button
                            className="confirm-button"
                            onClick={handleConfirm}
                            disabled={loading}
                        >
                            {loading ? 'Loading...' : 'Confirm Selection'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
