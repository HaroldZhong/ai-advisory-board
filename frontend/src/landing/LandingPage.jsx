import React, { useState, useEffect } from 'react';
import {
    Terminal,
    Github,
    ChevronRight,
    Check,
    Shield,
    Zap,
    Users,
    BarChart,
    Search,
    MessageSquare,
    Cpu,
    FileText,
    Brain,
    Lock,
    Code,
    Copy,
    Menu,
    X,
    Play,
    Loader2,
    Globe,
    Upload,
    Pencil,
    FolderOpen,
    Monitor,
    UserCog
} from 'lucide-react';
import { Button } from '../components/ui/button'; // Assuming these exist, otherwise I'd use standard buttons
import { Card, CardContent } from '../components/ui/card';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';

// Utility for smooth scroll
const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
};

export default function LandingPage() {
    const navigate = useNavigate();
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [activeStage, setActiveStage] = useState(0);
    const [promptText, setPromptText] = useState("");

    // Onboarding Configuration State
    const [showConfigModal, setShowConfigModal] = useState(false);
    const [apiKey, setApiKey] = useState('');
    const [isSaving, setIsSaving] = useState(false);
    const [configError, setConfigError] = useState('');

    // Auto-cycle the "How it works" or Hero animation
    useEffect(() => {
        const interval = setInterval(() => {
            setActiveStage((prev) => (prev + 1) % 3);
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    const handleActionClick = async () => {
        try {
            const status = await api.getConfigStatus();
            if (status.has_api_key) {
                navigate('/app');
            } else {
                setShowConfigModal(true);
            }
        } catch (e) {
            console.error("Failed to check config status:", e);
            setShowConfigModal(true);
        }
    };

    const handleSaveConfig = async () => {
        if (!apiKey.trim()) {
            setConfigError('API Key is required');
            return;
        }
        setIsSaving(true);
        setConfigError('');
        try {
            await api.setupConfig(apiKey.trim());
            setShowConfigModal(false);
            navigate('/app');
        } catch (e) {
            setConfigError(e.message || 'Failed to save configuration');
        } finally {
            setIsSaving(false);
        }
    };

    return (
        <div className="dark min-h-screen h-screen overflow-y-auto overflow-x-hidden bg-[#0A0C10] text-slate-50 font-sans selection:bg-indigo-500/30 scroll-smooth bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-900/10 via-[#0A0C10] to-[#0A0C10]">
            {/* 1) Top Navigation (Sticky) */}
            <nav className="glass-dark sticky top-0 z-50 w-full">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-2 font-bold text-xl tracking-tight">
                        <div className="w-8 h-8 flex items-center justify-center">
                            <img src="/favicon.png" alt="Logo" className="w-8 h-8 rounded-lg" />
                        </div>
                        <span>AI Advisory Board</span>
                    </div>

                    {/* Desktop Nav */}
                    <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-300">
                        {['How it works', 'Features', 'Demo', 'Use Cases', 'Quickstart', 'FAQ'].map((item) => (
                            <button
                                key={item}
                                onClick={() => scrollToSection(item.toLowerCase().replace(/\s+/g, '-'))}
                                className="hover:text-white transition-colors"
                            >
                                {item}
                            </button>
                        ))}
                    </div>

                    <div className="hidden md:flex items-center gap-4">
                        <Button variant="outline" className="border-slate-700 text-slate-300 hover:text-white hover:bg-slate-800" onClick={() => window.open('https://github.com/HaroldZhong/ai-advisory-board', '_blank')}>
                            <Github className="w-4 h-4 mr-2" /> GitHub
                        </Button>
                        <Button className="bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20" onClick={handleActionClick}>
                            <Terminal className="w-4 h-4 mr-2" /> Explore App
                        </Button>
                    </div>

                    {/* Mobile Menu Toggle */}
                    <button className="md:hidden p-2 text-slate-300" onClick={() => setIsMenuOpen(!isMenuOpen)}>
                        {isMenuOpen ? <X /> : <Menu />}
                    </button>
                </div>

                {/* Mobile Menu */}
                {isMenuOpen && (
                    <div className="md:hidden border-t border-white/10 bg-slate-950 p-4 flex flex-col gap-4">
                        {['How it works', 'Features', 'Demo', 'Use Cases', 'Quickstart', 'FAQ'].map((item) => (
                            <button
                                key={item}
                                onClick={() => {
                                    scrollToSection(item.toLowerCase().replace(/\s+/g, '-'));
                                    setIsMenuOpen(false);
                                }}
                                className="text-left text-slate-300 hover:text-white py-2"
                            >
                                {item}
                            </button>
                        ))}
                        <div className="flex flex-col gap-3 mt-4">
                            <Button variant="outline" className="w-full justify-start border-slate-700" onClick={() => window.open('https://github.com/HaroldZhong/ai-advisory-board', '_blank')}>
                                <Github className="w-4 h-4 mr-2" /> GitHub
                            </Button>
                            <Button className="w-full justify-start bg-indigo-600" onClick={() => scrollToSection('quickstart')}>
                                <Terminal className="w-4 h-4 mr-2" /> Run locally
                            </Button>
                        </div>
                    </div>
                )}
            </nav>

            {/* 2) Above-the-fold Hero */}
            <section className="relative pt-20 pb-20 md:pt-32 md:pb-32 overflow-hidden">
                {/* Background Gradients */}
                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-indigo-600/30 rounded-full blur-[120px] -z-10" />
                <div className="absolute top-1/4 -left-1/4 w-[600px] h-[600px] bg-purple-600/20 rounded-full blur-[100px] -z-10" />
                <div className="absolute bottom-0 right-0 w-[800px] h-[600px] bg-emerald-600/10 rounded-full blur-[100px] -z-10" />

                <div className="container mx-auto px-4 grid lg:grid-cols-2 gap-12 items-center">
                    <div className="max-w-2xl">
                        <h1 className="text-4xl md:text-6xl font-bold tracking-tight leading-tight mb-6">
                            Consensus answers from a <span className="text-gradient">council of LLMs</span>
                        </h1>
                        <p className="text-lg md:text-xl text-slate-400 mb-8 leading-relaxed">
                            Ask multiple models, let them review each other, then get a final synthesis with a confidence signal and retrieval-assisted context.
                        </p>

                        {/* Immediate Engagement Input */}
                        <div className="glass-card p-2 rounded-2xl flex flex-col sm:flex-row gap-2 mb-8 items-center shadow-indigo-500/10 shadow-2xl relative overflow-hidden group border-white/10">
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:animate-[shimmer_2s_infinite]" />
                            <div className="w-full flex items-center px-4">
                                <Search className="w-5 h-5 text-indigo-400 mr-3 shrink-0 group-focus-within:text-purple-400 transition-colors" />
                                <input
                                    type="text"
                                    value={promptText}
                                    onChange={(e) => setPromptText(e.target.value)}
                                    placeholder="Ask the council anything..."
                                    className="w-full bg-transparent border-none outline-none text-white placeholder:text-slate-500 py-3 text-lg focus:ring-0"
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') handleActionClick();
                                    }}
                                />
                            </div>
                            <Button
                                size="lg"
                                onClick={handleActionClick}
                                className="w-full sm:w-auto bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl h-12 shadow-lg shadow-indigo-600/20 shrink-0 transition-transform active:scale-95"
                            >
                                Convene Council <Zap className="w-4 h-4 ml-2" />
                            </Button>
                        </div>

                        <div className="flex flex-wrap items-center gap-4">
                            <Button size="sm" variant="outline" className="glass hover:bg-white/10 text-slate-300 border-white/10" onClick={handleActionClick}>
                                Open App Dashboard
                            </Button>
                            <button onClick={() => scrollToSection('how-it-works')} className="text-indigo-400 hover:text-indigo-300 font-medium text-sm flex items-center gap-1 px-4 transition-colors">
                                See how it works <ChevronRight className="w-4 h-4" />
                            </button>
                        </div>
                    </div>

                    {/* Hero Visual */}
                    <div className="relative group perspective-1000">
                        <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/20 via-purple-500/10 to-transparent rounded-3xl -z-10 blur-xl group-hover:blur-2xl transition-all duration-500" />
                        <div className="glass-card rounded-2xl p-6 shadow-2xl transition-transform duration-500 hover:scale-[1.02] border-white/5">
                            {/* Mock Header */}
                            <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/10">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-red-500/50" />
                                    <div className="w-3 h-3 rounded-full bg-amber-500/50" />
                                    <div className="w-3 h-3 rounded-full bg-emerald-500/50" />
                                </div>
                                <div className="text-xs font-mono text-slate-500">council_mode: active</div>
                            </div>

                            {/* Dynamic Mock Content */}
                            <div className="space-y-6">
                                <div className="flex gap-4">
                                    <div className="w-8 h-8 rounded bg-slate-800 flex-shrink-0" />
                                    <div className="space-y-2 flex-1">
                                        <div className="h-4 bg-slate-800 rounded w-3/4" />
                                        <div className="h-4 bg-slate-800 rounded w-1/2" />
                                    </div>
                                </div>

                                {/* Animated Stages */}
                                <div className="grid grid-cols-3 gap-4">
                                    {[0, 1, 2].map((i) => (
                                        <div key={i} className={`p-3 rounded-lg border transition-all duration-500 ${activeStage === i
                                            ? 'bg-indigo-950/50 border-indigo-500/50'
                                            : 'bg-slate-950 border-slate-800 opacity-50'
                                            }`}>
                                            <div className="flex items-center justify-between mb-2">
                                                <div className="text-[10px] font-bold text-slate-400">
                                                    {i === 0 ? 'STAGE 1: COLLECT' : i === 1 ? 'STAGE 2: RANK' : 'STAGE 3: SYNTH'}
                                                </div>
                                                {activeStage === i && <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />}
                                            </div>
                                            <div className="space-y-1">
                                                <div className="h-2 bg-slate-800 rounded w-full" />
                                                <div className="h-2 bg-slate-800 rounded w-2/3" />
                                            </div>
                                        </div>
                                    ))}
                                </div>

                                {/* Final Result Mock */}
                                <div className={`transition-opacity duration-500 ${activeStage === 2 ? 'opacity-100' : 'opacity-50'}`}>
                                    <div className="bg-emerald-950/20 border border-emerald-900/30 rounded-lg p-4">
                                        <div className="flex items-center justify-between mb-3">
                                            <span className="text-xs font-bold text-emerald-400 flex items-center gap-1">
                                                <Shield className="w-3 h-3" /> CONFIDENCE: HIGH
                                            </span>
                                            <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded-full text-slate-400">
                                                $0.002
                                            </span>
                                        </div>
                                        <div className="h-2 bg-slate-700 rounded w-full mb-2" />
                                        <div className="h-2 bg-slate-700 rounded w-5/6 mb-2" />
                                        <div className="h-2 bg-slate-700 rounded w-4/5" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* 3) Problem and Promise */}
            <section className="py-20 bg-slate-950">
                <div className="container mx-auto px-4 max-w-4xl text-center">
                    <h2 className="text-3xl md:text-4xl font-bold mb-12">Why a council beats a single guess</h2>

                    <div className="grid md:grid-cols-3 gap-8 text-left">
                        <div className="bg-slate-900/50 p-6 rounded-xl border border-white/5">
                            <div className="w-10 h-10 bg-red-500/10 rounded-lg flex items-center justify-center mb-4 text-red-400">
                                <X className="w-5 h-5" />
                            </div>
                            <p className="text-slate-300">Single-model chats can be <strong className="text-white">overconfident</strong> and hallucinate without earning it.</p>
                        </div>
                        <div className="bg-slate-900/50 p-6 rounded-xl border border-white/5">
                            <div className="w-10 h-10 bg-amber-500/10 rounded-lg flex items-center justify-center mb-4 text-amber-400">
                                <Search className="w-5 h-5" />
                            </div>
                            <p className="text-slate-300">Follow-up questions <strong className="text-white">lose useful context</strong> without retrieval support.</p>
                        </div>
                        <div className="bg-slate-900/50 p-6 rounded-xl border border-white/5">
                            <div className="w-10 h-10 bg-indigo-500/10 rounded-lg flex items-center justify-center mb-4 text-indigo-400">
                                <BarChart className="w-5 h-5" />
                            </div>
                            <p className="text-slate-300">Costs can <strong className="text-white">spike</strong> when there are no guardrails or visibility.</p>
                        </div>
                    </div>

                    <p className="mt-12 text-xl text-indigo-200 font-medium">
                        AI Advisory Board gives you diverse perspectives, transparent ranking, retrieval-assisted follow-ups, and cost insight.
                    </p>
                </div>
            </section>

            {/* 4) How It Works */}
            <section id="how-it-works" className="py-20 bg-slate-900/40 border-y border-white/5">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl font-bold mb-4">How it works</h2>
                        <p className="text-slate-400">Council Mode for hard questions. Chat Mode for fast follow-ups.</p>
                    </div>

                    <div className="relative grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
                        {/* Connecting Line (Desktop) */}
                        <div className="hidden md:block absolute top-12 left-[16%] right-[16%] h-0.5 bg-gradient-to-r from-indigo-900 via-indigo-500 to-indigo-900 border-t border-dashed border-indigo-500/30 -z-10" />

                        <div className="relative bg-slate-950 p-8 rounded-2xl border border-slate-800 z-10">
                            <div className="w-12 h-12 bg-slate-900 border border-slate-700 rounded-xl flex items-center justify-center mx-auto mb-6 text-xl font-bold text-slate-300">1</div>
                            <h3 className="text-xl font-semibold mb-3 text-center">Collect</h3>
                            <p className="text-slate-400 text-center text-sm">A curated or custom council answers independently so you can compare perspectives before synthesis.</p>
                        </div>

                        <div className="relative bg-slate-950 p-8 rounded-2xl border border-slate-800 z-10">
                            <div className="w-12 h-12 bg-slate-900 border border-slate-700 rounded-xl flex items-center justify-center mx-auto mb-6 text-xl font-bold text-slate-300">2</div>
                            <h3 className="text-xl font-semibold mb-3 text-center">Rank</h3>
                            <p className="text-slate-400 text-center text-sm">Models anonymously evaluate and rank each other's answers for quality and accuracy.</p>
                        </div>

                        <div className="relative bg-slate-950 p-8 rounded-2xl border border-slate-800 z-10">
                            <div className="w-12 h-12 bg-indigo-600 rounded-xl flex items-center justify-center mx-auto mb-6 text-xl font-bold text-white shadow-lg shadow-indigo-500/30">3</div>
                            <h3 className="text-xl font-semibold mb-3 text-center">Synthesize</h3>
                            <p className="text-slate-400 text-center text-sm">A Chairman model produces the final response plus a confidence indicator.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* 5) Benefits */}
            <section className="py-20 bg-slate-950">
                <div className="container mx-auto px-4 max-w-6xl">
                    <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {[
                            { icon: Shield, title: "More reliable answers", desc: "Reduces hallucinations via consensus.", sub: "Cross-model validation", grad: "from-blue-500/20 to-indigo-500/5" },
                            { icon: Search, title: "More transparent", desc: "See exactly how models ranked each other.", sub: "Full deliberation logs", grad: "from-purple-500/20 to-pink-500/5" },
                            { icon: MessageSquare, title: "Better continuity", desc: "Bring relevant prior context into long chats.", sub: "Retrieval support", grad: "from-emerald-500/20 to-teal-500/5" },
                            { icon: BarChart, title: "Predictable costs", desc: "Track spend per message and keep cost visible during the session.", sub: "Cost visibility", grad: "from-amber-500/20 to-orange-500/5" },
                        ].map((b, i) => (
                            <Card key={i} className={`glass-card relative overflow-hidden group border-white/5`}>
                                <div className={`absolute inset-0 bg-gradient-to-br ${b.grad} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
                                <CardContent className="p-6 relative z-10 pt-6">
                                    <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center mb-6 border border-white/10 group-hover:scale-110 group-hover:bg-white/10 transition-all duration-300">
                                        <b.icon className="w-6 h-6 text-indigo-400 group-hover:text-purple-300" />
                                    </div>
                                    <h3 className="text-xl font-semibold mb-3 text-slate-100">{b.title}</h3>
                                    <p className="text-sm text-slate-400 mb-6 leading-relaxed">{b.desc}</p>
                                    <div className="inline-block px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-xs font-medium text-indigo-300">
                                        {b.sub}
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </div>
            </section>

            {/* 6) Features Highlights */}
            <section id="features" className="py-20 bg-slate-900/20">
                <div className="container mx-auto px-4">
                    <h2 className="text-3xl font-bold mb-12 text-center">Advanced Capabilities</h2>
                    <div className="grid md:grid-cols-2 LG:grid-cols-3 gap-8">
                        {[
                            { title: "Configurable Councils", desc: "Compose a chairman and council from a curated OpenRouter model registry.", icon: Users },
                            { title: "Provider Transparency", desc: "Your OpenRouter key stays local and model/provider choices remain explicit.", icon: Lock },
                            { title: "Session Status", desc: "Current mode and session cost stay close to the chat workflow.", icon: Shield },
                            { title: "Local Key Setup", desc: "Connect your OpenRouter key locally before convening the council.", icon: Terminal },
                            { title: "Web Search", desc: "Perplexity-powered web search with fast and deep modes, inline toggle from the chat bar.", icon: Globe },
                            { title: "File Processing", desc: "Drag & drop PDF, DOCX, PPTX, XLSX, CSV, images and more for use in the turns where you attach them.", icon: Upload },
                            { title: "Edit & Regenerate", desc: "Click any previous message to edit and regenerate from that point.", icon: Pencil },
                            { title: "Simple Defaults", desc: "Auto-routing keeps common choices lightweight while advanced settings stay out of the way.", icon: UserCog },
                            { title: "Folder Organization", desc: "Group conversations into color-coded folders for easy management.", icon: FolderOpen },
                            { title: "Responsive Desktop", desc: "The app now works cleanly down to a 960x600 desktop window.", icon: Monitor },
                            { title: "Curated Models", desc: "Frontier and specialist options from OpenAI, Anthropic, Google, DeepSeek, Kimi, Qwen, xAI, Mistral, and more.", icon: Users },
                            { title: "Confidence Scoring", desc: "HIGH / MEDIUM / LOW based on council consensus and synthesis.", icon: Shield },
                            { title: "Persistent Memory", desc: "Local storage and retrieval support help long conversations stay easier to revisit.", icon: Brain },
                            { title: "Attachment Support", desc: "Uploaded files can be attached to requests and used as added context for that turn.", icon: FileText },
                            { title: "Cost Controls", desc: "Per-call usage and session cost visibility help you monitor spend while you work.", icon: BarChart },
                        ].map((f, i) => (
                            <div key={i} className="group p-6 rounded-xl bg-slate-950 border border-slate-800 hover:border-indigo-500/50 transition-all">
                                <div className="flex items-center gap-3 mb-4">
                                    <div className="w-10 h-10 rounded-lg bg-slate-900 flex items-center justify-center group-hover:bg-indigo-500/10 transition-colors">
                                        <f.icon className="w-5 h-5 text-slate-400 group-hover:text-indigo-400" />
                                    </div>
                                    <h3 className="font-semibold text-slate-200">{f.title}</h3>
                                </div>
                                <p className="text-sm text-slate-400">{f.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* 7) Demo Section */}
            <section id="demo" className="py-20 bg-slate-950 relative overflow-hidden">
                <div className="container mx-auto px-4 max-w-5xl">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl font-bold mb-4">See it in action</h2>
                        <p className="text-slate-400">Try these prompts to see the Council shine.</p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-12">
                        {/* Prompts */}
                        <div className="space-y-4">
                            {[
                                "Compare two approaches and justify the winner.",
                                "Summarize this document and call out uncertainties.",
                                "Answer, then explain confidence and what would change it."
                            ].map((prompt, i) => (
                                <div key={i} className="bg-slate-900 p-4 rounded-lg border border-slate-800 flex justify-between items-center group hover:border-indigo-500/50 cursor-pointer">
                                    <code className="text-sm text-slate-300 font-mono">{prompt}</code>
                                    <Button size="icon" variant="ghost" className="text-slate-500 hover:text-white" onClick={() => navigator.clipboard.writeText(prompt)}>
                                        <Copy className="w-4 h-4" />
                                    </Button>
                                </div>
                            ))}
                        </div>

                        {/* Visual Preview List */}
                        <div className="bg-slate-900/50 rounded-xl p-6 border border-white/5">
                            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-6">What you'll see</h3>
                            <ul className="space-y-4">
                                {[
                                    "Council membership before the first turn",
                                    "Side-by-side answers from multiple models",
                                    "Anonymous ranking & critique",
                                    "Final synthesis by Chairman",
                                    "Confidence badge (High/Med/Low)",
                                    "Conversation status and cost cues near the chat workflow"
                                ].map((item, i) => (
                                    <li key={i} className="flex items-center gap-3 text-slate-300 text-sm">
                                        <div className="w-5 h-5 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-500">
                                            <Check className="w-3 h-3" />
                                        </div>
                                        {item}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>
            </section>

            {/* 8) Use Cases */}
            <section id="use-cases" className="py-20 bg-slate-900/30">
                <div className="container mx-auto px-4">
                    <h2 className="text-3xl font-bold mb-12 text-center">Use Cases</h2>
                    <div className="grid md:grid-cols-3 gap-6">
                        {[
                            { icon: FileText, title: "Research & Writing", example: "Synthesize 3 papers into one survey." },
                            { icon: Cpu, title: "Engineering Decisions", example: "Compare PostgreSQL vs MongoDB for..." },
                            { icon: Brain, title: "Product Strategy", example: "Critique this feature spec for gaps." },
                            { icon: BarChart, title: "Data Analysis", example: "Explain this SQL query's edge cases." },
                            { icon: Users, title: "Learning & Tutoring", example: "Explain Quantum Computing like I'm 5." },
                            { icon: Shield, title: "Policy Review", example: "Check this clause for loopholes." },
                        ].map((u, i) => (
                            <div key={i} className="bg-slate-950 p-6 rounded-xl border border-slate-800 hover:border-slate-700 transition-colors">
                                <u.icon className="w-8 h-8 text-indigo-500 mb-4" />
                                <h3 className="font-semibold text-white mb-2">{u.title}</h3>
                                <p className="text-sm text-slate-400 italic">"{u.example}"</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* 9) Quickstart */}
            <section id="quickstart" className="py-20 bg-indigo-950/20 relative">
                <div className="absolute inset-0 bg-slate-950/80 -z-10" />
                <div className="container mx-auto px-4">
                    <div className="grid lg:grid-cols-2 gap-12 items-start">
                        <div>
                            <h2 className="text-3xl font-bold mb-6">Run it locally in minutes</h2>
                            <p className="text-slate-400 mb-8">Deploy the full council architecture on your machine. Keep your keys, data, and costs under your control.</p>

                            <div className="space-y-4 mb-8">
                                <div className="flex items-center gap-2 text-sm text-slate-300">
                                    <Check className="w-4 h-4 text-emerald-500" /> MIT Licensed
                                </div>
                                <div className="flex items-center gap-2 text-sm text-slate-300">
                                    <Check className="w-4 h-4 text-emerald-500" /> Python 3.10+ required
                                </div>
                            </div>

                            <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl">
                                <div className="flex items-center justify-between px-4 py-3 bg-slate-900 border-b border-slate-800">
                                    <div className="flex gap-1.5">
                                        <div className="w-3 h-3 rounded-full bg-red-500/50" />
                                        <div className="w-3 h-3 rounded-full bg-amber-500/50" />
                                        <div className="w-3 h-3 rounded-full bg-emerald-500/50" />
                                    </div>
                                    <span className="text-xs text-slate-500 font-mono">bash</span>
                                </div>
                                <pre className="p-4 text-sm font-mono text-slate-300 overflow-x-auto">
                                    {`# Clone the repository
git clone https://github.com/HaroldZhong/ai-advisory-board.git
cd ai-advisory-board

# Install dependencies (backend & frontend)
./start.ps1  # Windows
# or
./start.sh   # Mac/Linux

# Open http://localhost:5173`}
                                </pre>
                            </div>
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold mb-6">Architecture Mini-map</h3>
                            <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
                                <div className="space-y-6">
                                    <div className="flex items-start gap-4">
                                        <div className="mt-1 w-2 h-2 rounded-full bg-indigo-500" />
                                        <div>
                                            <h4 className="font-bold text-white text-sm">Frontend (React + Vite)</h4>
                                            <p className="text-xs text-slate-400 mt-1">Responsive UI, local key setup, chat status, cost visibility.</p>
                                        </div>
                                    </div>
                                    <div className="flex items-start gap-4">
                                        <div className="mt-1 w-2 h-2 rounded-full bg-emerald-500" />
                                        <div>
                                            <h4 className="font-bold text-white text-sm">Backend (FastAPI)</h4>
                                            <p className="text-xs text-slate-400 mt-1">Orchestration engine, tool management, RAG pipeline.</p>
                                        </div>
                                    </div>
                                    <div className="flex items-start gap-4">
                                        <div className="mt-1 w-2 h-2 rounded-full bg-amber-500" />
                                        <div>
                                            <h4 className="font-bold text-white text-sm">Data & Keys</h4>
                                            <p className="text-xs text-slate-400 mt-1">Your .env file, local file-based storage, local retrieval index.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* 10) Trust & FAQ */}
            <section id="faq" className="py-20 bg-slate-950">
                <div className="container mx-auto px-4 max-w-4xl">
                    <div className="grid md:grid-cols-2 gap-12 mb-20">
                        <div>
                            <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                                <Lock className="w-5 h-5 text-emerald-500" /> Trust & Privacy
                            </h3>
                            <ul className="space-y-3">
                                {[
                                    "Self-hostable local setup",
                                    "API key stays in your env variables",
                                    "Provider choices remain explicit",
                                    "Local key handling",
                                    "Attachment upload support",
                                    "Cost transparency during active sessions"
                                ].map((item, i) => (
                                    <li key={i} className="flex items-center gap-3 text-slate-300">
                                        <Check className="w-4 h-4 text-emerald-500" />
                                        {item}
                                    </li>
                                ))}
                            </ul>
                            <p className="mt-4 text-xs text-slate-500">* Costs depend on the models you select through OpenRouter.</p>
                        </div>

                        <div>
                            <h3 className="text-xl font-bold mb-6">Social Proof</h3>
                            <div className="flex items-center gap-4 mb-6">
                                <div className="px-4 py-2 bg-slate-900 rounded-full border border-slate-800 text-sm font-medium">
                                    ⭐ GitHub Stars (Coming Soon)
                                </div>
                            </div>
                            <div className="space-y-4">
                                <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 text-sm text-slate-400 italic">
                                    "Finally, a way to get a second opinion from AI without copy-pasting ten times."
                                    <div className="mt-2 text-xs not-italic text-indigo-400 font-bold">— ML Engineer</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <h2 className="text-3xl font-bold text-center mb-12">FAQ</h2>
                    <div className="space-y-4">
                        {[
                            { q: "What is Council Mode vs Chat Mode?", a: "Council Mode activates the 3-stage deliberation (Collect, Rank, Synthesize). Chat Mode is a standard single-model turn for fast follow-ups." },
                            { q: "How is confidence calculated?", a: "We aggregate the anonymous rankings from the 'Rank' stage and the Chairman's final synthesized assessment." },
                            { q: "What models/providers can I use?", a: "AI Advisory Board ships with a curated OpenRouter registry and live availability checks against OpenRouter's 350+ model catalog. Current examples include Claude Opus, GPT-5, Claude Sonnet, Gemini Pro, DeepSeek, Kimi, Qwen, xAI, and Mistral models." },
                            { q: "Does it support multi-turn context?", a: "Yes. Local conversation storage and retrieval-assisted history help preserve prior-turn context. Uploaded files are used when attached to a request." },
                            { q: "How do costs work?", a: "You bring your own OpenRouter key. The app tracks token usage per call and keeps session cost visible while you work." },
                            { q: "Can I self-host and control data?", a: "Yes. The stack runs locally and stores app data on your machine. Model calls and enabled provider features still route through the providers you choose." }
                        ].map((item, i) => (
                            <details key={i} className="group bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
                                <summary className="flex items-center justify-between p-4 cursor-pointer font-medium hover:bg-slate-800/50 transition-colors">
                                    {item.q}
                                    <ChevronRight className="w-4 h-4 transition-transform group-open:rotate-90" />
                                </summary>
                                <div className="p-4 pt-0 text-slate-400 text-sm leading-relaxed border-t border-slate-800/50 mt-2">
                                    {item.a}
                                </div>
                            </details>
                        ))}
                    </div>
                </div>
            </section>

            {/* 13) Final CTA */}
            <section className="py-24 bg-gradient-to-b from-slate-950 to-indigo-950/20 text-center">
                <div className="container mx-auto px-4">
                    <h2 className="text-4xl font-bold mb-8">Ready to convene your council?</h2>
                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-6">
                        <Button size="lg" className="bg-indigo-600 hover:bg-indigo-500 text-white h-14 px-8 text-lg w-full sm:w-auto" onClick={() => scrollToSection('quickstart')}>
                            <Terminal className="w-5 h-5 mr-3" /> Run locally
                        </Button>
                        <Button size="lg" variant="outline" className="border-slate-700 text-slate-200 hover:bg-slate-800 h-14 px-8 text-lg w-full sm:w-auto" onClick={() => window.open('https://github.com/HaroldZhong/ai-advisory-board', '_blank')}>
                            <Github className="w-5 h-5 mr-3" /> View on GitHub
                        </Button>
                    </div>
                    <p className="text-slate-400">Bring your OpenRouter key. Compose a council. Keep model choices and costs understandable.</p>
                </div>
            </section>

            {/* 14) Footer */}
            <footer className="py-12 bg-slate-950 border-t border-slate-900">
                <div className="container mx-auto px-4 flex flex-col md:flex-row items-center justify-between gap-6">
                    <div className="flex items-center gap-2 opacity-80">
                        <Brain className="w-5 h-5 text-indigo-500" />
                        <span className="font-bold text-slate-300">AI Advisory Board</span>
                    </div>

                    <div className="flex items-center gap-6 text-sm text-slate-500">
                        <a href="#" className="hover:text-white transition-colors">GitHub</a>
                        <a href="#" className="hover:text-white transition-colors">Docs</a>
                        <a href="#" className="hover:text-white transition-colors">License</a>
                    </div>

                    <div className="text-sm text-slate-600">
                        © {new Date().getFullYear()} AI Advisory Board. MIT License.
                    </div>
                </div>
            </footer>

            {/* Config / Onboarding Modal */}
            {showConfigModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm transition-all">
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 max-w-md w-full shadow-2xl relative animate-in fade-in zoom-in-95 duration-200">
                        <button
                            onClick={() => setShowConfigModal(false)}
                            className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
                        >
                            <X className="w-5 h-5" />
                        </button>

                        <div className="w-12 h-12 bg-indigo-500/10 rounded-xl flex items-center justify-center mb-6 border border-indigo-500/20">
                            <Lock className="w-6 h-6 text-indigo-400" />
                        </div>

                        <h2 className="text-2xl font-bold text-white mb-2">Welcome to the Board</h2>
                        <p className="text-slate-400 text-sm mb-6 leading-relaxed">
                            To convene the council, you'll need an <a href="https://openrouter.ai" target="_blank" rel="noreferrer" className="text-indigo-400 hover:underline">OpenRouter API Key</a>.
                            This key stays locally on your machine and gives you access to dozens of models.
                        </p>

                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                                    OpenRouter API Key
                                </label>
                                <input
                                    type="password"
                                    value={apiKey}
                                    onChange={(e) => setApiKey(e.target.value)}
                                    placeholder="sk-or-v1-..."
                                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-white placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all font-mono text-sm"
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') handleSaveConfig();
                                    }}
                                />
                                {configError && (
                                    <p className="text-red-400 text-xs mt-1">{configError}</p>
                                )}
                            </div>

                            <Button
                                onClick={handleSaveConfig}
                                disabled={isSaving || !apiKey.trim()}
                                className="w-full bg-indigo-600 hover:bg-indigo-500 text-white h-12 text-base font-medium transition-all"
                            >
                                {isSaving ? (
                                    <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Saving & Connecting...</>
                                ) : (
                                    'Connect & Continue'
                                )}
                            </Button>
                        </div>

                        <div className="mt-6 pt-6 border-t border-slate-800/50 text-center">
                            <p className="text-xs text-slate-500">
                                Don't have a key? <a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer" className="text-indigo-400 hover:underline">Get one for free</a>.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
