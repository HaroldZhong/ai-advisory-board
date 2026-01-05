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
    Play
} from 'lucide-react';
import { Button } from '../components/ui/button'; // Assuming these exist, otherwise I'd use standard buttons
import { Card, CardContent } from '../components/ui/card';

// Utility for smooth scroll
const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
};

export default function LandingPage() {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [activeStage, setActiveStage] = useState(0);

    // Auto-cycle the "How it works" or Hero animation
    useEffect(() => {
        const interval = setInterval(() => {
            setActiveStage((prev) => (prev + 1) % 3);
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="dark min-h-screen h-screen overflow-y-auto overflow-x-hidden bg-slate-950 text-slate-50 font-sans selection:bg-indigo-500/30 scroll-smooth">
            {/* 1) Top Navigation (Sticky) */}
            <nav className="sticky top-0 z-50 w-full border-b border-white/10 bg-slate-950/80 backdrop-blur-md">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-2 font-bold text-xl tracking-tight">
                        <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                            <Brain className="w-5 h-5 text-white" />
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
                        <Button className="bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20" onClick={() => scrollToSection('quickstart')}>
                            <Terminal className="w-4 h-4 mr-2" /> Run locally
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
                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-indigo-600/20 rounded-full blur-[120px] -z-10" />
                <div className="absolute bottom-0 right-0 w-[800px] h-[600px] bg-emerald-600/10 rounded-full blur-[100px] -z-10" />

                <div className="container mx-auto px-4 grid lg:grid-cols-2 gap-12 items-center">
                    <div className="max-w-2xl">
                        <h1 className="text-4xl md:text-6xl font-bold tracking-tight leading-tight mb-6">
                            Consensus answers from a <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-emerald-400">council of LLMs</span>
                        </h1>
                        <p className="text-lg md:text-xl text-slate-400 mb-8 leading-relaxed">
                            Ask multiple models, let them review each other, then get a final synthesis with a confidence signal and RAG-powered multi-turn context.
                        </p>

                        <ul className="space-y-3 mb-8 text-slate-300">
                            <li className="flex items-center gap-3">
                                <div className="w-6 h-6 rounded-full bg-indigo-500/20 flex items-center justify-center text-indigo-400">
                                    <Users className="w-4 h-4" />
                                </div>
                                Multi-model deliberation: collect, rank, synthesize
                            </li>
                            <li className="flex items-center gap-3">
                                <div className="w-6 h-6 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-400">
                                    <MessageSquare className="w-4 h-4" />
                                </div>
                                Multi-turn chat with advanced RAG
                            </li>
                            <li className="flex items-center gap-3">
                                <div className="w-6 h-6 rounded-full bg-amber-500/20 flex items-center justify-center text-amber-400">
                                    <BarChart className="w-4 h-4" />
                                </div>
                                Cost visibility: usage analytics per conversation
                            </li>
                        </ul>

                        <div className="flex flex-wrap items-center gap-4">
                            <Button size="lg" className="bg-indigo-600 hover:bg-indigo-500 text-white h-12 px-8 text-base shadow-xl shadow-indigo-600/20" onClick={() => scrollToSection('quickstart')}>
                                Run locally
                            </Button>
                            <Button size="lg" variant="outline" className="border-slate-700 text-slate-200 hover:bg-slate-800 h-12 px-6" onClick={() => window.open('https://github.com/HaroldZhong/ai-advisory-board', '_blank')}>
                                <Github className="w-5 h-5 mr-2" /> View on GitHub
                            </Button>
                            <button onClick={() => scrollToSection('how-it-works')} className="text-indigo-400 hover:text-indigo-300 font-medium text-sm flex items-center gap-1 px-4">
                                See how it works <ChevronRight className="w-4 h-4" />
                            </button>
                        </div>
                        <p className="mt-4 text-xs text-slate-500 font-mono">Works with OpenRouter API key</p>
                    </div>

                    {/* Hero Visual */}
                    <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/10 to-transparent rounded-2xl -z-10" />
                        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl backdrop-blur-sm">
                            {/* Mock Header */}
                            <div className="flex items-center justify-between mb-6 pb-4 border-b border-slate-800">
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
                            <p className="text-slate-300">Follow-up questions <strong className="text-white">lose context</strong> without advanced retrieval strategies.</p>
                        </div>
                        <div className="bg-slate-900/50 p-6 rounded-xl border border-white/5">
                            <div className="w-10 h-10 bg-indigo-500/10 rounded-lg flex items-center justify-center mb-4 text-indigo-400">
                                <BarChart className="w-5 h-5" />
                            </div>
                            <p className="text-slate-300">Costs can <strong className="text-white">spike</strong> when there are no guardrails or visibility.</p>
                        </div>
                    </div>

                    <p className="mt-12 text-xl text-indigo-200 font-medium">
                        AI Advisory Board gives you diverse perspectives, transparent ranking, context-aware follow-ups, and cost insight.
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
                            <p className="text-slate-400 text-center text-sm">Multiple LLMs (e.g., GPT-5.2, Claude Opus 4.5, Gemini 3 Pro Preview) answer your query independently.</p>
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
                            { icon: Shield, title: "More reliable answers", desc: "Reduces hallucinations via consensus.", sub: "Cross-model validation" },
                            { icon: Search, title: "More transparent decisions", desc: "See exactly how models ranked each other.", sub: "Full deliberation logs" },
                            { icon: MessageSquare, title: "Better multi-turn continuity", desc: "Never lose context in long chats.", sub: "RAG history injection" },
                            { icon: BarChart, title: "Predictable iteration costs", desc: "Track spend per message in real-time.", sub: "Token usage tracking" },
                        ].map((b, i) => (
                            <Card key={i} className="bg-slate-900/50 border-slate-800 hover:bg-slate-900 transition-colors">
                                <CardContent className="p-6">
                                    <b.icon className="w-8 h-8 text-indigo-500 mb-4" />
                                    <h3 className="text-lg font-semibold mb-2 text-slate-100">{b.title}</h3>
                                    <p className="text-sm text-slate-400 mb-3">{b.desc}</p>
                                    <p className="text-xs font-mono text-indigo-400/80 uppercase tracking-wider">{b.sub}</p>
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
                            { title: "Hybrid Retrieval", desc: "BM25 keyword + dense semantic search, fused for better recall.", icon: Search },
                            { title: "Query Rewriting", desc: "Follow-ups rewritten to resolve pronouns and references before searching.", icon: MessageSquare },
                            { title: "Confidence Scoring", desc: "HIGH / MEDIUM / LOW based on council consensus and synthesis.", icon: Shield },
                            { title: "Chain-of-thought", desc: "Show reasoning steps for models that support it.", icon: Brain },
                            { title: "Cost Tracking", desc: "Real-time usage and cost analytics per conversation.", icon: BarChart },
                            { title: "Model Selection", desc: "Pick council members and chairman dynamically.", icon: Users },
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
                                    "Side-by-side answers from multiple models",
                                    "Anonymous ranking & critique",
                                    "Final synthesis by Chairman",
                                    "Confidence badge (High/Med/Low)",
                                    "Real-time cost summary"
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
                                            <p className="text-xs text-slate-400 mt-1">Responsive UI, real-time streaming, cost visualization.</p>
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
                                            <p className="text-xs text-slate-400 mt-1">Your .env file, local chromaDB vector store.</p>
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
                                    "Cost transparency per session",
                                    "Confidence indicator for uncertainty"
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
                            { q: "What models/providers can I use?", a: "We support OpenRouter, so you can use almost any major model (GPT-5.2, Claude Opus 4.5, Gemini 3 Pro Preview, etc.)." },
                            { q: "Does it support multi-turn context?", a: "Yes. We use advanced RAG with query rewriting to maintain context across long conversations." },
                            { q: "How do costs work?", a: "You bring your own API key. We track token usage per call and estimate costs in real-time." },
                            { q: "Can I self-host and control data?", a: "Absolutely. The entire stack runs locally on your machine. No data is sent to our servers." }
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
                    <p className="text-slate-400">Bring your OpenRouter key. Choose your models. Start debating.</p>
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
        </div>
    );
}
